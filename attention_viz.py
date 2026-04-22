# ============================================================
# attention_viz.py
# Attention visualisation + early convergence analysis
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import gzip

os.makedirs('results', exist_ok=True)

# ---- Same config as day_1_test.py ----
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
TAPE_LEN   = 64
N1D_WIN    = 2
NCA1D_VOCAB = N_STATES ** N1D_WIN           # 4
N_RULES    = 50
N_TRAJ     = 200
N_STEPS    = 32
NCA_EPOCHS = 15
DYCK_VOCAB = 4
DYCK_DEPTH = 8
DYCK_SEQ_LEN = 64
N_DYCK_TRAIN = 10_000
N_DYCK_VAL   = 2_000
DYCK_EPOCHS  = 15    # we only need 15 epochs here, not 80
BATCH_SIZE   = 32
NCA_LR       = 3e-4
DYCK_LR      = 1e-4
SEEDS        = [42, 43, 44]
TOK_OPEN     = 0
TOK_CLOSE    = 1
TOK_PAD      = 2
TOK_EOS      = 3

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CausalSelfAttention(nn.Module):
    """
    Same as day_1_test.py but with return_attn flag.
    When return_attn=True, returns (output, attn_weights).
    attn_weights shape: (B, n_heads, T, T)
    """
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

        # attn is the matrix we want to visualise
        # shape: (B, n_heads, T, T)
        # attn[b, h, i, j] = how much position i attends to position j
        attn = F.softmax(scores, dim=-1)
        out  = attn @ v
        out  = out.transpose(1, 2).contiguous().view(B, T, C)
        out  = self.out_proj(out)

        if return_attn:
            return out, attn   # caller gets both
        return out


class TransformerBlock(nn.Module):
    """
    Same as day_1_test.py but threads return_attn through.
    """
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
            return x, attn_weights   # pass weights up
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x


class TinyTransformer(nn.Module):
    """
    Same as day_1_test.py but with get_attention() method.
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed     = None
        self.head      = None
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm_out  = nn.LayerNorm(d_model)

    def set_task(self, vocab_size):
        self.embed = nn.Embedding(vocab_size, self.d_model)
        self.head  = nn.Linear(self.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, dtype=torch.long)
        x    = self.embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_out(x))

    def get_attention(self, idx, layer=0):
        """
        Run a forward pass and return attention weights from one layer.

        Args:
            idx   : (1, T) integer token tensor — ONE sequence, no batch
            layer : which transformer block to extract from (0 = first)

        Returns:
            attn_weights: (n_heads, T, T) numpy array
                          attn_weights[h, i, j] = head h's attention
                          from position i to position j
        """
        assert self.embed is not None, "Call set_task() first"
        self.eval()

        with torch.no_grad():
            B, T = idx.shape
            pos  = torch.arange(T, dtype=torch.long)
            x    = self.embed(idx) + self.pos_embed(pos)

            for i, block in enumerate(self.blocks):
                if i == layer:
                    # Extract attention weights from this specific block
                    x, attn_weights = block(x, return_attn=True)
                else:
                    x = block(x)

        # attn_weights: (B, n_heads, T, T) → squeeze batch dim
        return attn_weights.squeeze(0).cpu().numpy()   # (n_heads, T, T)

    def get_transferable_state(self):
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith('blocks') or
               k.startswith('pos_embed') or
               k.startswith('norm_out')
        }

    def load_transferable_state(self, state):
        self.load_state_dict(state, strict=False)

def make_2d_rule_network(n_states: int, hidden: int = 16, seed: int = 0) -> dict:
    """
    Sample a random NCA rule network.
    Architecture: Linear(9, hidden) → ReLU → Linear(hidden, n_states)
    Input: 9 values (cell + 8 Moore neighbours), each in [0, n_states-1]
    Output: logits over n_states next states

    Returns a dict of numpy arrays (not a PyTorch module — we run this
    on numpy directly because it's fast and avoids torch overhead in
    a tight generation loop).
    """
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / 9)           # Xavier for fan_in=9
    scale2 = np.sqrt(2.0 / hidden)
    return {
        'W1': rng.normal(0, scale1, (hidden, 9)).astype(np.float32),
        'b1': np.zeros(hidden, dtype=np.float32),
        'W2': rng.normal(0, scale2, (n_states, hidden)).astype(np.float32),
        'b2': np.zeros(n_states, dtype=np.float32),
    }


def apply_2d_rule(grid: np.ndarray, rule: dict) -> np.ndarray:
    """
    Apply one NCA timestep to the entire grid.

    grid : (H, W) integer array, values in [0, n_states-1]
    rule : dict from make_2d_rule_network
    returns: (H, W) next grid state
    """
    H, W = grid.shape
    g = grid.astype(np.float32)

    # Gather Moore neighbourhood for every cell using np.roll
    # np.roll wraps edges → toroidal (periodic) boundary conditions
    neighbourhood = np.stack([
        g,                                                          # self
        np.roll(g, -1, axis=0),                                     # N
        np.roll(np.roll(g, -1, axis=0),  1, axis=1),               # NE
        np.roll(g,  1, axis=1),                                     # E
        np.roll(np.roll(g,  1, axis=0),  1, axis=1),               # SE
        np.roll(g,  1, axis=0),                                     # S
        np.roll(np.roll(g,  1, axis=0), -1, axis=1),               # SW
        np.roll(g, -1, axis=1),                                     # W
        np.roll(np.roll(g, -1, axis=0), -1, axis=1),               # NW
    ], axis=-1).reshape(-1, 9)                                      # (H*W, 9)

    # Forward pass: Linear → ReLU → Linear → argmax
    h      = np.maximum(neighbourhood @ rule['W1'].T + rule['b1'], 0)  # (H*W, hidden)
    logits = h @ rule['W2'].T + rule['b2']                             # (H*W, n_states)
    return np.argmax(logits, axis=-1).reshape(H, W)                    # (H, W)

def tokenise_2d_grid(grid: np.ndarray, n_states: int, patch: int) -> list:
    """
    Convert a 2D grid into a flat list of integer tokens.

    Steps:
    1. Divide grid into non-overlapping (patch × patch) patches
    2. Encode each patch as a single integer in base n_states
       (like reading a number in base n_states)
    3. Read patches in raster order (row by row, left to right)

    Example (patch=2, n_states=4):
      patch = [[1, 2],   → token = 1*4^3 + 2*4^2 + 0*4^1 + 3*4^0 = 64+32+0+3 = 99
               [0, 3]]
    """
    H, W    = grid.shape
    tokens  = []
    for i in range(0, H, patch):
        for j in range(0, W, patch):
            p = grid[i:i+patch, j:j+patch].flatten()  # (patch*patch,) integers
            # Base-n encoding: p[0]*n^(k-1) + p[1]*n^(k-2) + ... + p[k-1]*n^0
            token = 0
            for val in p:
                token = token * n_states + int(val)
            tokens.append(token)
    return tokens


def generate_2d_nca_sequences(n_rules: int,
                               n_traj : int,
                               n_steps: int,
                               n_states: int = N_STATES,
                               H: int = GRID_H,
                               W: int = GRID_W,
                               patch: int = PATCH,
                               base_seed: int = 0,
                               complexity_filter=None) -> list:
    """
    Generate all NCA training sequences.

    For each rule:
      - Run n_traj different random initial states through the rule
      - For each trajectory, tokenise 2 consecutive steps → 128 tokens
      - This gives n_rules * n_traj sequences total

    complexity_filter: if provided, a tuple (low, high) — only keep rules
    whose gzip ratio falls in [low, high]. Used for curriculum condition D.

    Returns: list of (128,) numpy integer arrays
    """
    tokens_per_grid = (H // patch) * (W // patch)   # 16/2 * 16/2 = 64
    sequences = []
    rng = np.random.default_rng(base_seed)

    for rule_idx in range(n_rules):
        rule_seed = int(rng.integers(0, 1_000_000))
        rule = make_2d_rule_network(n_states=n_states, seed=rule_seed)

        # Complexity filter — compute gzip ratio of one sample trajectory
        if complexity_filter is not None:
            lo, hi = complexity_filter
            # Generate a sample trajectory to measure complexity
            sample_grid = rng.integers(0, n_states, size=(H, W))
            traj_bytes  = bytearray()
            g = sample_grid.copy()
            for _ in range(n_steps):
                g = apply_2d_rule(g, rule)
                traj_bytes.extend(g.flatten().astype(np.uint8).tobytes())
            compressed = gzip.compress(bytes(traj_bytes))
            ratio = len(compressed) / len(traj_bytes)
            if not (lo <= ratio <= hi):   # outside complexity band → skip
                continue

        # Generate n_traj sequences from different starting states
        for traj_idx in range(n_traj):
            traj_seed = int(rng.integers(0, 1_000_000))
            traj_rng  = np.random.default_rng(traj_seed)
            grid = traj_rng.integers(0, n_states, size=(H, W))

            # Run trajectory and collect tokenised steps
            step_tokens = []
            for _ in range(n_steps):
                grid = apply_2d_rule(grid, rule)
                step_tokens.append(tokenise_2d_grid(grid, n_states, patch))

            # Take just the last two steps — one sequence per trajectory
            # (using all consecutive pairs gives 63x too many sequences)
            seq = step_tokens[-2] + step_tokens[-1]  # 64 + 64 = 128 tokens
            sequences.append(np.array(seq[:SEQ_LEN], dtype=np.int64))

    return sequences

def generate_dyck1_sequence(max_depth: int, max_len: int, rng) -> list:
    """
    Generate one valid Dyck-1 sequence (balanced parentheses).
    """
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


def make_dyck_dataset(n_sequences: int, max_depth: int = DYCK_DEPTH,
                      max_len: int = DYCK_SEQ_LEN, seed: int = 0) -> torch.Tensor:
    """
    Generate a dataset of Dyck-1 sequences, padded to SEQ_LEN.
    Returns: (n_sequences, SEQ_LEN) long tensor
    """
    rng  = np.random.default_rng(seed)
    data = []
    for _ in range(n_sequences):
        seq = generate_dyck1_sequence(max_depth, max_len, rng)
        # Pad or truncate to SEQ_LEN
        seq = seq[:SEQ_LEN]
        seq = seq + [TOK_PAD] * (SEQ_LEN - len(seq))
        data.append(seq)
    return torch.tensor(data, dtype=torch.long)



def make_dataloader(sequences: list, batch_size: int, shuffle: bool = True):
    """
    Wrap a list of (SEQ_LEN,) arrays into a simple batch iterator.
    Yields (input, target) pairs where target = input shifted left by 1
    (standard next-token prediction).
    """
    data = torch.tensor(np.stack(sequences), dtype=torch.long)  # (N, SEQ_LEN)
    n    = len(data)
    idx  = list(range(n))
    if shuffle:
        random.shuffle(idx)

    for start in range(0, n - batch_size + 1, batch_size):
        batch = data[idx[start:start + batch_size]]  # (B, SEQ_LEN)
        # Input: all but last token
        # Target: all but first token (shifted by 1)
        x = batch[:, :-1]   # (B, SEQ_LEN-1)
        y = batch[:, 1:]    # (B, SEQ_LEN-1)
        yield x, y

def train_one_epoch(model: TinyTransformer,
                    sequences: list,
                    optimizer: torch.optim.Optimizer,
                    ignore_index: int = -1) -> float:
    """
    Train for one epoch. Returns mean loss.
    ignore_index: token index to ignore in loss (used to mask PAD tokens).
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y in make_dataloader(sequences, BATCH_SIZE, shuffle=True):
        optimizer.zero_grad()

        logits = model(x)                        # (B, T-1, vocab)
        # Flatten for cross-entropy: (B*(T-1), vocab) vs (B*(T-1),)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=ignore_index            # don't penalise padding
        )

        loss.backward()
        # Gradient clipping — prevents exploding gradients at small scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: TinyTransformer, data: torch.Tensor) -> float:
    """
    Compute mean validation loss on a dataset tensor.
    data: (N, SEQ_LEN) long tensor
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    sequences = [data[i].numpy() for i in range(len(data))]
    for x, y in make_dataloader(sequences, BATCH_SIZE, shuffle=False):
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=TOK_PAD   # ignore PAD tokens in val loss
        )
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)        

def train_and_checkpoint(condition: str,
                         nca_seqs_2d: list,
                         dyck_train: torch.Tensor,
                         dyck_val: torch.Tensor,
                         seed: int = 42,
                         checkpoint_epochs: list = [0, 5]) -> dict:
    """
    Train a model (scratch or NCA-pretrained) and save checkpoints
    at specified epochs.

    Args:
        condition         : 'A' (scratch) or 'B' (2D NCA)
        nca_seqs_2d       : pre-generated NCA sequences
        dyck_train/val    : Dyck-1 datasets
        seed              : random seed
        checkpoint_epochs : which fine-tuning epochs to snapshot

    Returns:
        checkpoints: dict mapping epoch → model state_dict
                     e.g. {0: state_at_epoch_0, 5: state_at_epoch_5}
    """
    set_seed(seed)
    model = TinyTransformer()
    checkpoints = {}

    # ---- Stage 0: NCA pre-pre-training (condition B only) ----
    if condition == 'B':
        print(f"  Pre-pre-training on 2D NCA ({NCA_EPOCHS} epochs)...")
        model.set_task(NCA_VOCAB)
        optimizer = torch.optim.Adam(model.parameters(), lr=NCA_LR)
        for epoch in range(NCA_EPOCHS):
            loss = train_one_epoch(model, nca_seqs_2d, optimizer)
        print(f"  Final NCA loss: {loss:.4f}")
        transferable = model.get_transferable_state()

    # ---- Switch to Dyck task ----
    model.set_task(DYCK_VOCAB)
    optimizer = torch.optim.Adam(model.parameters(), lr=DYCK_LR)

    if condition == 'B':
        model.load_transferable_state(transferable)

    # ---- Save epoch 0 checkpoint (before any Dyck training) ----
    if 0 in checkpoint_epochs:
        checkpoints[0] = {
            k: v.clone() for k, v in model.state_dict().items()
        }
        val_loss = evaluate(model, dyck_val)
        print(f"  Epoch 0  | val_loss={val_loss:.4f}  (checkpoint saved)")

    # ---- Fine-tune on Dyck, saving checkpoints ----
    dyck_seqs = [dyck_train[i].numpy() for i in range(len(dyck_train))]
    max_epoch  = max(checkpoint_epochs)

    for epoch in range(1, max_epoch + 1):
        train_one_epoch(model, dyck_seqs, optimizer, ignore_index=TOK_PAD)

        if epoch in checkpoint_epochs:
            checkpoints[epoch] = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            val_loss = evaluate(model, dyck_val)
            print(f"  Epoch {epoch:2d} | val_loss={val_loss:.4f}  (checkpoint saved)")

    return checkpoints

def make_probe_sequence():
    """
    Hand-craft a Dyck-1 sequence with clear matching structure.

    Sequence:  ( ( ) ( ( ) ) )
    Positions: 0 1 2 3 4 5 6 7

    Matching pairs:
      position 0 '(' matches position 7 ')'
      position 1 '(' matches position 2 ')'
      position 3 '(' matches position 6 ')'
      position 4 '(' matches position 5 ')'

    When predicting ')' at position 2, model SHOULD attend to '(' at 1.
    When predicting ')' at position 5, model SHOULD attend to '(' at 4.
    When predicting ')' at position 6, model SHOULD attend to '(' at 3.
    When predicting ')' at position 7, model SHOULD attend to '(' at 0.

    Returns:
        tokens    : (1, SEQ_LEN) long tensor (padded)
        token_labels: list of string labels for x/y axis ticks
    """
    # ( ( ) ( ( ) ) )  EOS  PAD PAD ...
    raw = [
        TOK_OPEN,   # 0  (
        TOK_OPEN,   # 1  (
        TOK_CLOSE,  # 2  )
        TOK_OPEN,   # 3  (
        TOK_OPEN,   # 4  (
        TOK_CLOSE,  # 5  )
        TOK_CLOSE,  # 6  )
        TOK_CLOSE,  # 7  )
        TOK_EOS,    # 8  EOS
    ]
    padded = raw + [TOK_PAD] * (SEQ_LEN - len(raw))
    tokens = torch.tensor([padded], dtype=torch.long)   # (1, SEQ_LEN)

    # Labels for the first 9 positions (the ones we'll show)
    labels = ['(', '(', ')', '(', '(', ')', ')', ')', 'EOS']

    return tokens, labels


# Annotate which positions are the correct matches for each ')'
# Used to draw red boxes on the heatmap showing "ground truth"
MATCHING_PAIRS = {
    2: 1,   # ')' at pos 2 should attend to '(' at pos 1
    5: 4,   # ')' at pos 5 should attend to '(' at pos 4
    6: 3,   # ')' at pos 6 should attend to '(' at pos 3
    7: 0,   # ')' at pos 7 should attend to '(' at pos 0
}

def plot_attention_heatmap(ax, model, tokens, token_labels,
                           layer=0, head='mean',
                           title='', show_seq_len=9):
    """
    Plot attention weights as a heatmap on a given matplotlib axis.

    Args:
        ax           : matplotlib axis to draw on
        model        : TinyTransformer with set_task() called
        tokens       : (1, SEQ_LEN) long tensor
        token_labels : list of string labels (e.g. ['(', '(', ')', ...])
        layer        : which transformer block to extract from
        head         : 'mean' (average all heads) or int (specific head)
        title        : subplot title
        show_seq_len : how many positions to show (crop the rest)
    """
    # Extract attention weights
    attn = model.get_attention(tokens, layer=layer)
    # attn: (n_heads, T, T)

    # Average across heads or select one
    if head == 'mean':
        attn_map = attn.mean(axis=0)   # (T, T)
    else:
        attn_map = attn[head]           # (T, T)

    # Crop to the interesting positions
    attn_map = attn_map[:show_seq_len, :show_seq_len]

    # Plot heatmap
    im = ax.imshow(attn_map, cmap='Blues', vmin=0, vmax=attn_map.max(),
                   aspect='auto', interpolation='nearest')

    # Axis labels
    ax.set_xticks(range(show_seq_len))
    ax.set_yticks(range(show_seq_len))
    ax.set_xticklabels(token_labels[:show_seq_len], fontsize=9, color='white')
    ax.set_yticklabels(token_labels[:show_seq_len], fontsize=9, color='white')
    ax.set_xlabel('Attending TO →', fontsize=9, color='white')
    ax.set_ylabel('← FROM position', fontsize=9, color='white')
    ax.set_title(title, fontsize=11, color='white', pad=8)
    ax.tick_params(colors='white')

    # Draw red boxes at the "correct" matching positions
    # For each ')' position, highlight where it SHOULD attend
    for close_pos, open_pos in MATCHING_PAIRS.items():
        if close_pos < show_seq_len and open_pos < show_seq_len:
            rect = plt.Rectangle(
                (open_pos - 0.5, close_pos - 0.5),  # (x, y)
                1, 1,                                 # width, height
                linewidth=2, edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    return im

def make_attention_comparison_figure(scratch_checkpoints: dict,
                                     nca_checkpoints: dict):
    """
    Build the 2x2 attention comparison figure.

              Epoch 0 (before Dyck)    Epoch 5 (after 5 epochs)
    Scratch   [heatmap]                [heatmap]
    NCA-init  [heatmap]                [heatmap]

    Red boxes = correct bracket-matching positions.
    If NCA heatmaps are brighter at red boxes → NCA installed structure.
    """
    tokens, labels = make_probe_sequence()

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'Attention Patterns: NCA Pre-Training Installs Bracket-Matching Structure\n'
        'Red boxes = correct matching bracket positions',
        fontsize=13, color='white', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    panel_configs = [
        # (row, col, checkpoints_dict, epoch, title)
        (0, 0, scratch_checkpoints, 0, 'Scratch — Epoch 0\n(random init, no Dyck training)'),
        (0, 1, scratch_checkpoints, 5, 'Scratch — Epoch 5\n(5 epochs of Dyck fine-tuning)'),
        (1, 0, nca_checkpoints,     0, '2D NCA Pre-trained — Epoch 0\n(after NCA, before Dyck)'),
        (1, 1, nca_checkpoints,     5, '2D NCA Pre-trained — Epoch 5\n(5 epochs of Dyck fine-tuning)'),
    ]

    for row, col, checkpoints, epoch, title in panel_configs:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0d0d1a')

        # Load the checkpoint into a fresh model
        model = TinyTransformer()
        model.set_task(DYCK_VOCAB)
        model.load_state_dict(checkpoints[epoch])

        plot_attention_heatmap(
            ax, model, tokens, labels,
            layer=0, head='mean',
            title=title,
            show_seq_len=9
        )

    plt.savefig('results/attention_comparison.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print('Saved: results/attention_comparison.png')
    plt.close()    

def make_zoom_plot(all_curves: dict):
    """
    Plot epochs 1-20 only, with AUC annotation.
    all_curves: dict mapping condition label -> list of per-seed loss lists
                (same format as day_1_test.py all_curves)
    """
    conditions = list(all_curves.keys())
    colours = {
        'A: Scratch':                    '#888888',
        'B: 2D NCA (paper)':             '#4dabf7',
        'C: 1D NCA (ours)':              '#69db7c',
        'D: 2D NCA + Curriculum (ours)': '#ffa94d',
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    zoom_end  = 20   # show only first 20 epochs
    auc_end   = 15   # compute AUC over first 15 epochs
    epochs    = list(range(1, zoom_end + 1))

    auc_values = {}

    for label, curves in all_curves.items():
        curves_arr = np.array(curves)                  # (n_seeds, n_epochs)
        mean = curves_arr.mean(axis=0)[:zoom_end]
        std  = curves_arr.std(axis=0)[:zoom_end]
        colour = colours.get(label, '#ffffff')

        ax.plot(epochs, mean, color=colour, label=label, linewidth=2.5)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=colour, alpha=0.2)

        # AUC = sum of losses for epochs 1..auc_end
        # Lower AUC = faster convergence = better
        auc_values[label] = curves_arr[:, :auc_end].mean(axis=0).sum()

    # Annotate AUC relative to scratch
    scratch_auc = auc_values.get('A: Scratch', 1.0)
    print("\nEarly Convergence (AUC, epochs 1-15, lower is better):")
    for label, auc in auc_values.items():
        reduction = (scratch_auc - auc) / scratch_auc * 100
        sign = '-' if reduction >= 0 else '+'
        print(f"  {label:<40} AUC={auc:.4f}  ({sign}{abs(reduction):.1f}% vs scratch)")

    ax.set_xlim(1, zoom_end)
    ax.set_xlabel('Fine-tuning epoch', fontsize=12, color='white')
    ax.set_ylabel('Validation loss', fontsize=12, color='white')
    ax.set_title(
        'Early Convergence: Epochs 1-20\n'
        'NCA pre-training advantage is visible in the warmup phase',
        fontsize=13, color='white'
    )
    ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig('results/early_convergence.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print('Saved: results/early_convergence.png')
    plt.close()    

def bracket_attention_score(model, tokens):
    """
    Quantify how much attention each ')' places on its correct '(' match.

    For each (close_pos, open_pos) pair in MATCHING_PAIRS, we measure
    attn[h, close_pos, open_pos] across all heads and average.

    Returns a single float.
    Random baseline = 1 / show_seq_len ≈ 0.11 for 9 positions.
    Higher = model is attending to the right places.
    """
    attn = model.get_attention(tokens, layer=0)  # (n_heads, T, T)
    scores = []
    for close_pos, open_pos in MATCHING_PAIRS.items():
        for head in range(attn.shape[0]):
            scores.append(float(attn[head, close_pos, open_pos]))
    return np.mean(scores)


def plot_all_heads(scratch_checkpoints, nca_checkpoints, epoch=0):
    """
    Plot all 4 attention heads side-by-side for scratch vs NCA at a given epoch.
    Saves: results/heads_epoch{epoch}.png

    Layout: 2 rows (Scratch, NCA) x 4 cols (Head 0-3)
    Reveals whether any individual head shows bracket-matching structure
    that the averaged heatmap was hiding.
    """
    tokens, labels = make_probe_sequence()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f'Individual Attention Heads — Epoch {epoch}\n'
        f'Top: Scratch | Bottom: NCA Pre-trained | Red boxes = correct matching positions',
        fontsize=13, color='white', y=0.98
    )

    row_configs = [
        (0, scratch_checkpoints, 'Scratch'),
        (1, nca_checkpoints,     '2D NCA Pre-trained'),
    ]

    for row, checkpoints, label in row_configs:
        # Load checkpoint into fresh model
        model = TinyTransformer()
        model.set_task(DYCK_VOCAB)
        model.load_state_dict(checkpoints[epoch])

        # Compute bracket score for this model
        score = bracket_attention_score(model, tokens)

        for head in range(N_HEADS):
            ax = axes[row, head]
            ax.set_facecolor('#0d0d1a')
            plot_attention_heatmap(
                ax, model, tokens, labels,
                layer=0, head=head,
                title=f'{label}\nHead {head}  (epoch {epoch})',
                show_seq_len=9
            )

        # Add bracket score annotation on the leftmost panel
        axes[row, 0].set_ylabel(
            f'{label}\nBracket score: {score:.3f}\n← FROM position',
            fontsize=9, color='white'
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'results/heads_epoch{epoch}.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved: results/heads_epoch{epoch}.png')
    plt.close()


def make_bracket_score_chart(scratch_checkpoints, nca_checkpoints):
    """
    Bar chart comparing bracket attention scores across:
      - Scratch vs NCA
      - Epoch 0 vs Epoch 5
      - Layer 0 vs Layer 1

    Saves: results/bracket_scores.png

    The random baseline is 1/show_seq_len = 1/9 ≈ 0.111.
    A score above baseline means the model attends to correct
    positions more than chance — even before seeing brackets.
    """
    tokens, _ = make_probe_sequence()
    show_seq_len = 9
    random_baseline = 1.0 / show_seq_len

    # Compute scores for all combinations
    conditions = {
        'Scratch\nEpoch 0':  (scratch_checkpoints, 0),
        'Scratch\nEpoch 5':  (scratch_checkpoints, 5),
        'NCA\nEpoch 0':      (nca_checkpoints,     0),
        'NCA\nEpoch 5':      (nca_checkpoints,     5),
    }

    # Store scores per layer
    scores_layer0 = {}
    scores_layer1 = {}

    for label, (checkpoints, epoch) in conditions.items():
        model = TinyTransformer()
        model.set_task(DYCK_VOCAB)
        model.load_state_dict(checkpoints[epoch])

        # Layer 0
        attn0 = model.get_attention(tokens, layer=0)
        s0 = []
        for close_pos, open_pos in MATCHING_PAIRS.items():
            for h in range(attn0.shape[0]):
                s0.append(float(attn0[h, close_pos, open_pos]))
        scores_layer0[label] = np.mean(s0)

        # Layer 1
        attn1 = model.get_attention(tokens, layer=1)
        s1 = []
        for close_pos, open_pos in MATCHING_PAIRS.items():
            for h in range(attn1.shape[0]):
                s1.append(float(attn1[h, close_pos, open_pos]))
        scores_layer1[label] = np.mean(s1)

    # ---- Print results ----
    print('\nBracket Attention Scores (higher = better, random baseline = {:.3f}):'.format(
        random_baseline))
    print(f'  {"Condition":<20} {"Layer 0":>10} {"Layer 1":>10}')
    print('  ' + '-' * 42)
    for label in conditions:
        clean = label.replace('\n', ' ')
        print(f'  {clean:<20} {scores_layer0[label]:>10.4f} {scores_layer1[label]:>10.4f}')

    # ---- Plot ----
    labels  = list(conditions.keys())
    x       = np.arange(len(labels))
    width   = 0.35
    colours = ['#4dabf7', '#ffa94d']   # blue=layer0, orange=layer1

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    bars0 = ax.bar(x - width/2,
                   [scores_layer0[l] for l in labels],
                   width, label='Layer 0', color=colours[0], alpha=0.85)
    bars1 = ax.bar(x + width/2,
                   [scores_layer1[l] for l in labels],
                   width, label='Layer 1', color=colours[1], alpha=0.85)

    # Random baseline
    ax.axhline(random_baseline, color='white', linestyle='--',
               linewidth=1.5, alpha=0.6,
               label=f'Random baseline ({random_baseline:.3f})')

    # Value labels on bars
    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=8, color='white')

    # Highlight NCA bars with a subtle box
    ax.axvspan(1.5, 3.5, alpha=0.06, color='#69db7c', label='NCA conditions')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color='white')
    ax.set_ylabel('Bracket Attention Score', fontsize=12, color='white')
    ax.set_title(
        'How much does the model attend to correct matching brackets?\n'
        'Above the dashed line = better than random chance',
        fontsize=12, color='white'
    )
    ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, max(max(scores_layer0.values()),
                       max(scores_layer1.values())) * 1.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig('results/bracket_scores.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print('Saved: results/bracket_scores.png')
    plt.close()


def main():
    print('=' * 60)
    print('attention_viz.py — Attention Visualisation')
    print('=' * 60)

    # ---- 1. Generate data ----
    print('\n[1/4] Generating NCA sequences...')
    # paste generate_2d_nca_sequences() from day_1_test.py
    nca_seqs_2d = generate_2d_nca_sequences(
        n_rules=N_RULES, n_traj=N_TRAJ, n_steps=N_STEPS, base_seed=0
    )
    print(f'  {len(nca_seqs_2d):,} sequences')

    print('[2/4] Generating Dyck-1 dataset...')
    dyck_train = make_dyck_dataset(N_DYCK_TRAIN, seed=100)
    dyck_val   = make_dyck_dataset(N_DYCK_VAL,   seed=101)

    # ---- 2. Train models and save checkpoints ----
    print('\n[3/4] Training models...')
    print('\n--- Condition A: Scratch ---')
    scratch_checkpoints = train_and_checkpoint(
        condition='A',
        nca_seqs_2d=nca_seqs_2d,
        dyck_train=dyck_train,
        dyck_val=dyck_val,
        seed=42,
        checkpoint_epochs=[0, 5]
    )

    print('\n--- Condition B: 2D NCA Pre-trained ---')
    nca_checkpoints = train_and_checkpoint(
        condition='B',
        nca_seqs_2d=nca_seqs_2d,
        dyck_train=dyck_train,
        dyck_val=dyck_val,
        seed=42,
        checkpoint_epochs=[0, 5]
    )

    # ---- 3. Make figures ----
    print('\n[4/4] Generating figures...')
    make_attention_comparison_figure(scratch_checkpoints, nca_checkpoints)

    # For the zoom plot, reconstruct all_curves from the results
    # you already have from day_1_test.py.
    # Paste your results here manually:
    all_curves = {
        'A: Scratch': [
            # seed 42 val losses epochs 1-80
            # seed 43 val losses epochs 1-80
            # seed 44 val losses epochs 1-80
            # (copy from day_1_test.py output)
        ],
        'B: 2D NCA (paper)': [],
        'C: 1D NCA (ours)':  [],
        'D: 2D NCA + Curriculum (ours)': [],
    }
    # If all_curves is populated, uncomment:
    # make_zoom_plot(all_curves)
    # Otherwise the attention figure alone is enough for today

    # ---- 4. Individual heads plot ----
    print('\n[5/5] Generating individual heads plots...')
    plot_all_heads(scratch_checkpoints, nca_checkpoints, epoch=0)
    plot_all_heads(scratch_checkpoints, nca_checkpoints, epoch=5)

    # ---- 5. Bracket attention score bar chart ----
    print('Generating bracket attention score chart...')
    make_bracket_score_chart(scratch_checkpoints, nca_checkpoints)

    print('\nDone. Check results/ folder.')


if __name__ == '__main__':
    main()