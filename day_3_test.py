# ============================================================
# day_3_test.py — The Catastrophic Forgetting Fix
#
# Finding from day_2/attention_viz:
#   NCA pre-training installs strong Layer 1 structure (score 0.246)
#   — exactly where induction heads form.
#   BUT fine-tuning destroys it: 0.246 → 0.164 after just 5 epochs.
#   This is catastrophic forgetting at toy scale.
#
# Hypothesis:
#   If we PROTECT the transferred circuits during fine-tuning
#   (lower LR or frozen attention), the head start should persist
#   long enough to produce measurable loss curve separation.
#
# Four conditions (stripped down — A and B are reference only):
#   A — Scratch baseline
#   B — 2D NCA, standard LR (previous result — known to forget)
#   E — 2D NCA, 10× lower LR on transferred layers (slow forgetting)
#   F — 2D NCA, frozen attention layers (no forgetting)
#
# Also tracks bracket attention score at epochs 0, 5, 20, 40, 80
# to see whether E/F actually preserve the Layer 1 structure.
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gzip
import random
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs('results', exist_ok=True)

# ============================================================
# CONFIG — same as day_1_test.py (keep in sync)
# ============================================================

D_MODEL    = 64
N_HEADS    = 4
D_FF       = 256
N_LAYERS   = 2
SEQ_LEN    = 256
GRID_H     = 16
GRID_W     = 16
N_STATES   = 2
PATCH      = 2
NCA_VOCAB  = N_STATES ** (PATCH * PATCH)   # 16
N_RULES    = 50
N_TRAJ     = 200
N_STEPS    = 32
NCA_EPOCHS = 15

DYCK_VOCAB    = 4
DYCK_DEPTH    = 8
DYCK_SEQ_LEN  = 64
N_DYCK_TRAIN  = 10_000
N_DYCK_VAL    = 2_000
DYCK_EPOCHS   = 80

BATCH_SIZE    = 32
NCA_LR        = 3e-4
DYCK_LR       = 1e-4
DYCK_LR_SLOW  = 1e-5    # 10× lower — for transferred layers in Condition E

SEEDS         = [42, 43, 44]
THRESHOLD     = 0.3

# Epochs at which to record bracket attention score
PROBE_EPOCHS  = [0, 5, 20, 40, 80]

TOK_OPEN      = 0
TOK_CLOSE     = 1
TOK_PAD       = 2
TOK_EOS       = 3

# Hand-crafted probe sequence for bracket attention scoring
# ( ( ) ( ( ) ) )  — same as attention_viz.py
MATCHING_PAIRS = {2: 1, 5: 4, 6: 3, 7: 0}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# MODEL — with return_attn support for probing
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scale  = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        mask   = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = attn @ v
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        out    = self.out_proj(out)

        if return_attn:
            return out, attn
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=SEQ_LEN):
        super().__init__()
        self.d_model   = d_model
        self.max_len   = max_len
        self.embed     = None
        self.head      = None
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm_out  = nn.LayerNorm(d_model)

    def set_task(self, vocab_size: int):
        self.embed = nn.Embedding(vocab_size, self.d_model)
        self.head  = nn.Linear(self.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx):
        assert self.embed is not None
        B, T = idx.shape
        pos  = torch.arange(T, dtype=torch.long)
        x    = self.embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_out(x))

    def get_attention(self, idx, layer=0):
        """Extract attention weights from one layer. Returns (n_heads, T, T)."""
        assert self.embed is not None
        self.eval()
        with torch.no_grad():
            B, T = idx.shape
            pos  = torch.arange(T, dtype=torch.long)
            x    = self.embed(idx) + self.pos_embed(pos)
            for i, block in enumerate(self.blocks):
                if i == layer:
                    x, attn_weights = block(x, return_attn=True)
                else:
                    x = block(x)
        return attn_weights.squeeze(0).cpu().numpy()

    def get_transferable_state(self) -> dict:
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith('blocks') or
               k.startswith('pos_embed') or
               k.startswith('norm_out')
        }

    def load_transferable_state(self, state: dict):
        self.load_state_dict(state, strict=False)

    def freeze_attention(self):
        """
        Freeze all attention parameters in all blocks.
        Only embed, head, MLPs, and LayerNorms remain trainable.

        WHY: If catastrophic forgetting is overwriting NCA-installed
        attention circuits, freezing them prevents this completely.
        The model can still learn Dyck via its MLP layers —
        which handle local pattern recognition — while the pre-trained
        attention handles structural context.
        """
        for block in self.blocks:
            for param in block.attn.parameters():
                param.requires_grad = False
        frozen = sum(
            p.numel() for block in self.blocks
            for p in block.attn.parameters()
        )
        print(f"  Frozen {frozen:,} attention params")

    def make_layerwise_optimizer(self, new_lr: float, transferred_lr: float):
        """
        Create an optimizer with different LRs for new vs transferred layers.

        New layers (embed, head):       new_lr          (e.g. 1e-4)
        Transferred layers (blocks):    transferred_lr  (e.g. 1e-5)

        WHY: The NCA-installed circuits in Layer 1 are valuable but fragile.
        A 10× lower LR on transferred layers means fine-tuning adapts them
        slowly, preserving the structural priors while still allowing
        task-specific refinement.
        """
        new_params         = []
        transferred_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('embed') or name.startswith('head'):
                new_params.append(param)
            else:
                transferred_params.append(param)

        optimizer = torch.optim.Adam([
            {'params': new_params,         'lr': new_lr},
            {'params': transferred_params, 'lr': transferred_lr},
        ])

        print(f"  Layerwise optimizer: "
              f"{len(new_params)} new param groups @ lr={new_lr}, "
              f"{len(transferred_params)} transferred param groups @ lr={transferred_lr}")
        return optimizer


# ============================================================
# DATA GENERATORS (copied from day_1_test.py)
# ============================================================

def make_2d_rule_network(n_states, hidden=16, seed=0):
    rng    = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / 9)
    scale2 = np.sqrt(2.0 / hidden)
    return {
        'W1': rng.normal(0, scale1, (hidden, 9)).astype(np.float32),
        'b1': np.zeros(hidden, dtype=np.float32),
        'W2': rng.normal(0, scale2, (n_states, hidden)).astype(np.float32),
        'b2': np.zeros(n_states, dtype=np.float32),
    }


def apply_2d_rule(grid, rule):
    H, W = grid.shape
    g = grid.astype(np.float32)
    neighbourhood = np.stack([
        g,
        np.roll(g, -1, axis=0),
        np.roll(np.roll(g, -1, axis=0),  1, axis=1),
        np.roll(g,  1, axis=1),
        np.roll(np.roll(g,  1, axis=0),  1, axis=1),
        np.roll(g,  1, axis=0),
        np.roll(np.roll(g,  1, axis=0), -1, axis=1),
        np.roll(g, -1, axis=1),
        np.roll(np.roll(g, -1, axis=0), -1, axis=1),
    ], axis=-1).reshape(-1, 9)
    h      = np.maximum(neighbourhood @ rule['W1'].T + rule['b1'], 0)
    logits = h @ rule['W2'].T + rule['b2']
    return np.argmax(logits, axis=-1).reshape(H, W)


def tokenise_2d_grid(grid, n_states, patch):
    H, W   = grid.shape
    tokens = []
    for i in range(0, H, patch):
        for j in range(0, W, patch):
            p     = grid[i:i+patch, j:j+patch].flatten()
            token = 0
            for val in p:
                token = token * n_states + int(val)
            tokens.append(token)
    return tokens


def generate_2d_nca_sequences(n_rules, n_traj, n_steps,
                               n_states=N_STATES, H=GRID_H, W=GRID_W,
                               patch=PATCH, base_seed=0):
    sequences = []
    rng = np.random.default_rng(base_seed)
    for _ in range(n_rules):
        rule_seed = int(rng.integers(0, 1_000_000))
        rule      = make_2d_rule_network(n_states=n_states, seed=rule_seed)
        for _ in range(n_traj):
            traj_seed = int(rng.integers(0, 1_000_000))
            grid      = np.random.default_rng(traj_seed).integers(
                            0, n_states, size=(H, W))
            step_tokens = []
            for _ in range(n_steps):
                grid = apply_2d_rule(grid, rule)
                step_tokens.append(tokenise_2d_grid(grid, n_states, patch))
            seq = step_tokens[-2] + step_tokens[-1]
            sequences.append(np.array(seq[:SEQ_LEN], dtype=np.int64))
    return sequences


def generate_dyck1_sequence(max_depth, max_len, rng):
    tokens = []
    depth  = 0
    while len(tokens) < max_len - depth:
        if depth == 0:
            tokens.append(TOK_OPEN);  depth += 1
        elif depth >= max_depth:
            tokens.append(TOK_CLOSE); depth -= 1
        else:
            if rng.random() < 0.5:
                tokens.append(TOK_OPEN);  depth += 1
            else:
                tokens.append(TOK_CLOSE); depth -= 1
    tokens.extend([TOK_CLOSE] * depth)
    tokens.append(TOK_EOS)
    return tokens


def make_dyck_dataset(n_sequences, max_depth=DYCK_DEPTH,
                      max_len=DYCK_SEQ_LEN, seed=0):
    rng  = np.random.default_rng(seed)
    data = []
    for _ in range(n_sequences):
        seq = generate_dyck1_sequence(max_depth, max_len, rng)
        seq = seq[:SEQ_LEN]
        seq = seq + [TOK_PAD] * (SEQ_LEN - len(seq))
        data.append(seq)
    return torch.tensor(data, dtype=torch.long)


def make_dataloader(sequences, batch_size, shuffle=True):
    data = torch.tensor(np.stack(sequences), dtype=torch.long)
    idx  = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(data) - batch_size + 1, batch_size):
        batch = data[idx[start:start + batch_size]]
        yield batch[:, :-1], batch[:, 1:]


def train_one_epoch(model, sequences, optimizer, ignore_index=-1):
    model.train()
    total, n = 0.0, 0
    for x, y in make_dataloader(sequences, BATCH_SIZE, shuffle=True):
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=ignore_index
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    total, n = 0.0, 0
    sequences = [data[i].numpy() for i in range(len(data))]
    for x, y in make_dataloader(sequences, BATCH_SIZE, shuffle=False):
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=TOK_PAD
        )
        total += loss.item(); n += 1
    return total / max(n, 1)


# ============================================================
# BRACKET ATTENTION PROBE
# ============================================================

def make_probe_tokens():
    """( ( ) ( ( ) ) ) EOS PAD..."""
    raw    = [TOK_OPEN, TOK_OPEN, TOK_CLOSE, TOK_OPEN, TOK_OPEN,
              TOK_CLOSE, TOK_CLOSE, TOK_CLOSE, TOK_EOS]
    padded = raw + [TOK_PAD] * (SEQ_LEN - len(raw))
    return torch.tensor([padded], dtype=torch.long)


def bracket_score(model, tokens):
    """
    Mean attention at correct bracket-matching positions,
    averaged across all heads and both layers.
    Higher = more structured. Random baseline = 1/9 ≈ 0.111.
    Returns (layer0_score, layer1_score).
    """
    scores = {}
    for layer in [0, 1]:
        attn = model.get_attention(tokens, layer=layer)  # (n_heads, T, T)
        vals = []
        for close_pos, open_pos in MATCHING_PAIRS.items():
            for h in range(attn.shape[0]):
                vals.append(float(attn[h, close_pos, open_pos]))
        scores[layer] = np.mean(vals)
    return scores[0], scores[1]


# ============================================================
# RUN ONE CONDITION
# ============================================================

def run_condition(condition, seed, nca_seqs, dyck_train, dyck_val):
    """
    Run one condition × seed.

    condition:
      'A' — scratch (no pre-pre-training, standard LR)
      'B' — 2D NCA pre-trained, standard LR (previous result)
      'E' — 2D NCA pre-trained, 10× lower LR on transferred layers
      'F' — 2D NCA pre-trained, attention layers frozen during fine-tuning

    Returns:
      val_losses       : list of float, one per epoch
      bracket_scores   : dict mapping epoch → (layer0_score, layer1_score)
      steps_to_threshold: int or None
    """
    set_seed(seed)
    model       = TinyTransformer()
    probe_tokens = make_probe_tokens()

    # ---- Stage 0: NCA Pre-Pre-Training ----
    if condition in ('B', 'E', 'F'):
        print(f"  [Seed {seed}] {condition}: NCA pre-pre-training ({NCA_EPOCHS} epochs)...")
        model.set_task(NCA_VOCAB)
        nca_opt = torch.optim.Adam(model.parameters(), lr=NCA_LR)
        for ep in range(NCA_EPOCHS):
            loss = train_one_epoch(model, nca_seqs, nca_opt)
        print(f"    NCA loss: {loss:.4f}")
        transferable = model.get_transferable_state()

    # ---- Switch to Dyck task ----
    model.set_task(DYCK_VOCAB)

    if condition in ('B', 'E', 'F'):
        model.load_transferable_state(transferable)

    # ---- Condition F: freeze attention before fine-tuning ----
    if condition == 'F':
        model.freeze_attention()

    # ---- Build optimizer ----
    if condition == 'E':
        # Layerwise LR: new layers at 1e-4, transferred at 1e-5
        optimizer = model.make_layerwise_optimizer(
            new_lr=DYCK_LR, transferred_lr=DYCK_LR_SLOW
        )
    else:
        # Standard: all trainable params at same LR
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=DYCK_LR
        )

    # ---- Fine-tune with probing ----
    val_losses        = []
    attn_scores       = {}
    steps_to_threshold = None
    dyck_seqs         = [dyck_train[i].numpy() for i in range(len(dyck_train))]

    # Probe at epoch 0 (before any Dyck training)
    if 0 in PROBE_EPOCHS:
        l0, l1 = bracket_score(model, probe_tokens)
        attn_scores[0] = (l0, l1)

    print(f"  [Seed {seed}] {condition}: Fine-tuning on Dyck-1 ({DYCK_EPOCHS} epochs)...")
    for epoch in range(1, DYCK_EPOCHS + 1):
        train_one_epoch(model, dyck_seqs, optimizer, ignore_index=TOK_PAD)
        val_loss = evaluate(model, dyck_val)
        val_losses.append(val_loss)

        if steps_to_threshold is None and val_loss < THRESHOLD:
            steps_to_threshold = epoch

        # Probe at specified epochs
        if epoch in PROBE_EPOCHS:
            l0, l1 = bracket_score(model, probe_tokens)
            attn_scores[epoch] = (l0, l1)

        if epoch % 10 == 0:
            probe_str = ''
            if epoch in attn_scores:
                l0, l1 = attn_scores[epoch]
                probe_str = f'  | L0={l0:.3f} L1={l1:.3f}'
            print(f"    Epoch {epoch:3d} | val={val_loss:.4f}{probe_str}")

    final = val_losses[-1]
    steps = steps_to_threshold or f">{DYCK_EPOCHS}"
    print(f"  → Final: {final:.4f} | Threshold: {steps}")
    return val_losses, attn_scores, steps_to_threshold


# ============================================================
# PLOTTING
# ============================================================

def plot_loss_curves(all_curves, filename='results/day3_loss_curves.png'):
    colours = {
        'A: Scratch':               '#888888',
        'B: NCA standard LR':       '#4dabf7',
        'E: NCA slow LR (ours)':    '#69db7c',
        'F: NCA frozen attn (ours)':'#ffa94d',
    }
    epochs = list(range(1, DYCK_EPOCHS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'Catastrophic Forgetting Fix: Protecting NCA-Installed Circuits\n'
        'Left: full run | Right: epochs 1-20 (warmup)',
        fontsize=13, color='white'
    )

    for ax, xlim, title in [
        (axes[0], (1, DYCK_EPOCHS), 'Full training run'),
        (axes[1], (1, 20),          'Warmup (epochs 1-20)'),
    ]:
        ax.set_facecolor('#1a1a2e')
        for label, curves in all_curves.items():
            arr  = np.array(curves)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            colour = colours.get(label, '#ffffff')
            ax.plot(epochs, mean, color=colour, label=label, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=colour, alpha=0.2)
        ax.set_xlim(*xlim)
        ax.set_xlabel('Epoch', color='white', fontsize=11)
        ax.set_ylabel('Val loss', color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=11)
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved: {filename}')
    plt.close()


def plot_bracket_scores_over_time(all_scores, filename='results/day3_bracket_scores.png'):
    """
    Line chart showing bracket attention score (Layer 1) over training epochs.
    Key question: do conditions E and F maintain higher scores than B?
    """
    colours = {
        'A: Scratch':               '#888888',
        'B: NCA standard LR':       '#4dabf7',
        'E: NCA slow LR (ours)':    '#69db7c',
        'F: NCA frozen attn (ours)':'#ffa94d',
    }
    random_baseline = 1.0 / 9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'Bracket Attention Score Over Training\n'
        'Does protecting NCA circuits preserve the Layer 1 advantage?',
        fontsize=13, color='white'
    )

    for ax_idx, (ax, layer, title) in enumerate(zip(
        axes,
        [0, 1],
        ['Layer 0', 'Layer 1 (induction heads)']
    )):
        ax.set_facecolor('#1a1a2e')
        ax.axhline(random_baseline, color='white', linestyle='--',
                   alpha=0.5, label=f'Random ({random_baseline:.3f})')

        for label, scores_per_seed in all_scores.items():
            # scores_per_seed: list of dicts (epoch → (l0, l1))
            probe_epochs = sorted(scores_per_seed[0].keys())
            vals = []
            for ep in probe_epochs:
                seed_vals = [s[ep][layer] for s in scores_per_seed]
                vals.append(np.mean(seed_vals))

            colour = colours.get(label, '#ffffff')
            ax.plot(probe_epochs, vals, color=colour,
                    label=label, linewidth=2, marker='o', markersize=5)

        ax.set_xlabel('Fine-tuning epoch', color='white', fontsize=11)
        ax.set_ylabel('Bracket attention score', color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=11)
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved: {filename}')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print('=' * 65)
    print('day_3_test.py — Catastrophic Forgetting Fix')
    print('=' * 65)

    # ---- Generate data once ----
    print('\n[1/3] Generating 2D NCA sequences...')
    nca_seqs = generate_2d_nca_sequences(
        n_rules=N_RULES, n_traj=N_TRAJ, n_steps=N_STEPS, base_seed=0
    )
    print(f'  {len(nca_seqs):,} sequences')

    print('[2/3] Generating Dyck-1 dataset...')
    dyck_train = make_dyck_dataset(N_DYCK_TRAIN, seed=100)
    dyck_val   = make_dyck_dataset(N_DYCK_VAL,   seed=101)
    print(f'  Train: {len(dyck_train):,} | Val: {len(dyck_val):,}')

    # ---- Run all conditions × seeds ----
    print('\n[3/3] Running experiments...')
    conditions = ['A', 'B', 'E', 'F']
    labels = {
        'A': 'A: Scratch',
        'B': 'B: NCA standard LR',
        'E': 'E: NCA slow LR (ours)',
        'F': 'F: NCA frozen attn (ours)',
    }

    all_curves  = defaultdict(list)   # label → list of loss arrays
    all_scores  = defaultdict(list)   # label → list of score dicts
    all_results = defaultdict(list)   # label → list of (final_loss, steps)

    for cond in conditions:
        lbl = labels[cond]
        print(f'\n{"="*55}')
        print(f'Condition {cond}: {lbl}')
        print(f'{"="*55}')
        for seed in SEEDS:
            val_losses, attn_scores, steps = run_condition(
                cond, seed, nca_seqs, dyck_train, dyck_val
            )
            all_curves[lbl].append(val_losses)
            all_scores[lbl].append(attn_scores)
            all_results[lbl].append((val_losses[-1], steps))

    # ---- Results table ----
    print('\n' + '=' * 65)
    print('RESULTS SUMMARY')
    print('=' * 65)
    print(f'{"Condition":<35} {"Final Loss":>12} {"Steps to <0.3":>15}')
    print('-' * 65)

    for lbl in [labels[c] for c in conditions]:
        losses = [r[0] for r in all_results[lbl]]
        steps  = [r[1] for r in all_results[lbl] if r[1] is not None]
        mean_l = np.mean(losses)
        std_l  = np.std(losses)
        mean_s = np.mean(steps) if steps else float('nan')
        s_str  = f'{mean_s:.0f}' if not np.isnan(mean_s) else f'>{DYCK_EPOCHS}'
        print(f'{lbl:<35} {mean_l:.4f} ± {std_l:.4f}   {s_str:>10}')

    # ---- Bracket score table ----
    print('\nBracket Attention Scores at key epochs (Layer 1, mean across seeds):')
    print(f'{"Condition":<35}', end='')
    for ep in PROBE_EPOCHS:
        print(f'  ep{ep:>2}', end='')
    print()
    print('-' * 65)

    random_baseline = 1.0 / 9
    print(f'{"Random baseline":<35}', end='')
    for ep in PROBE_EPOCHS:
        print(f'  {random_baseline:.3f}', end='')
    print()

    for lbl in [labels[c] for c in conditions]:
        print(f'{lbl:<35}', end='')
        for ep in PROBE_EPOCHS:
            vals = [s[ep][1] for s in all_scores[lbl] if ep in s]
            mean = np.mean(vals) if vals else float('nan')
            print(f'  {mean:.3f}', end='')
        print()

    # ---- Plots ----
    print('\nGenerating plots...')
    plot_loss_curves(all_curves)
    plot_bracket_scores_over_time(all_scores)

    elapsed = time.time() - t_start
    print(f'\nTotal runtime: {elapsed/60:.1f} minutes')
    print('=' * 65)
    print('Results in: results/day3_loss_curves.png')
    print('           results/day3_bracket_scores.png')


if __name__ == '__main__':
    main()