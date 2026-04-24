#--------------------------
# day_1_test.py - Test Run
# Target: < 5 min on CPU
#---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gzip
import random 
import time 
import matplotlib
matplotlib.use('Agg') # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt 
from collections import defaultdict 

#--------------------------------
# Configuration - Hyperparameters
# --------------------------------

# --- Model Architecture ---
D_MODEL  = 64   # embedding dimension - small but not trivial 
N_HEADS  = 4    # attention heads - D_MODEL must be divisible by N_HEADS 
D_FF     = 256  #feed-forward hidden dimensions 
N_LAYERS = 2    # transformer blocks - 2 layers 
SEQ_LEN  = 256  # was 128 — more temporal context to infer rules 

# --- 2D NCA Parameters --- 
GRID_H   = 16   # grid height (paper uses 12)
GRID_W   = 16   # grid width 
N_STATES = 2    # was 4 — smaller vocab (2^4=16 for 2D, same as 1D) so model can actually learn
PATCH    = 2    # patch size - 2X2 patches

#vocab size = N_STATES^(PATCH*PATCH)  = 4^(2*2)=256 
NCA_VOCAB = N_STATES ** (PATCH * PATCH) #256

# --- 1D NCA data generation --- 
TAPE_LEN = 64   # length of 1D tape 
N1D_WIN  = 2    # window size for tokenization ( 2 cells -> 1 token) 
#vocab size = N_STATES^(N1D_WIN)  = 4^(2)=16 
NCA1D_VOCAB = N_STATES ** N1D_WIN #16


#--- NCA data generation ---
N_RULES   = 50  # was 500 — fewer rules, learn them deeply
N_TRAJ    = 200 # was 20  — more trajectories per rule (forces in-context learning)
N_STEPS   = 32  # was 64  — keep reasonable, faster generation
NCA_EPOCHS = 15 # was 3   — actually converge the NCA task 

# --- Transfer task (DYCK-1) --- 
# Vocabulary: 0='(', 1=')', 2=PAD, 3=EOS 
DYCK_VOCAB   = 4 
DYCK_DEPTH   = 8  # max nesting depth 
DYCK_SEQ_LEN = 64 # length before padding 
N_DYCK_TRAIN = 10_000 
N_DYCK_VAL   = 2_000 
DYCK_EPOCHS  = 80  # was 20 — run to convergence, gap shows up early 

# --- Training --- 
BATCH_SIZE   = 32 
NCA_LR       = 3e-4
DYCK_LR      = 1e-4

# --- Experiments --- 
SEEDS = [42, 43, 44] # 3 seeds -> means (+/-) std 
THRESHOLD = 0.3 # val loss threshold for "steps to convergence" metric 

print(f"NCA 2D vocab size : {NCA_VOCAB}")
print(f"NCA 1D vocab size : {NCA1D_VOCAB}") 
print(f"Dyck vocab size : {DYCK_VOCAB}") 
print(f"Total sequences : {N_RULES * N_TRAJ:,}") 

def set_seed(seed: int): 
    """
    Fix all random sources so results are reproducible.
    Call this at the start of each condition x seed combination. 
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed) 

class CausalSelfAttention(nn.Module): 
    """
    Multi-head causal self attention. 
    'Causal' - each position can only attend to earilier positions
    no peeking at future tokens. Makes next-token prediction work.
    """

    def __init__(self, d_model:int, n_heads:int): 
        super().__init__() 
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads" 

        self.n_heads = n_heads 
        self.d_head = d_model // n_heads # dimension per head 

        # Q, K, V projections - fused into one matrix for efficiency 
        self.qkv_proj = nn.Linear(d_model, 3*d_model, bias=False) 
        self.out_proj = nn.Linear(d_model, d_model, bias=False) 

    def forward (self, x):
        B, T, C = x.shape #batch, sequence length, d_model 

        # Project to Q, K, V and split into heads 
        qkv = self.qkv_proj(x)  #(B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1) # each (B, T, C) 

        # Reshape to (B, n_heads, T, d_head) 
        def split_heads(t): 
            return t.view(B, T, self.n_heads, self.d_head).transpose(1,2) 

        q, k, v = split_heads(q), split_heads(k), split_heads(v) 

        # Scaled dot-product attention with causal mask 
        # scale by 1/sqrt(d_head) to prevent softmax saturation 
        scale = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Causal mask: prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf')) 

        att = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        out = att @ v   # (B, n_heads, T, d_head)

        # Merge heads back 
        out = out.transpose(1,2).contiguous().view(B,T,C) #(B,T,C) 
        return self.out_proj(out) 

class TransformerBlock(nn.Module): 
    """
    One transformer decoder block: 
    LayerNorm -> Attention -> residual 
    LayerNorm -> FFN -> residual 

    Pre-norm (norm before attention) is more stable than post-norm
    for small models and short training runs. 
    """ 
    def __init__( self, d_model:int, n_heads:int, d_ff:int):
        super().__init__() 
        self.norm1 =nn.LayerNorm(d_model) 
        self.attn = CausalSelfAttention(d_model, n_heads) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )  

    def forward(self, x): 
        x = x+self.attn(self.norm1(x)) #attention with residual 
        x = x + self.ff(self.norm2(x))
        return x 


class TinyTransformer(nn.Module):
    """
    ~150k parameter transformer with a clean transfer interface.

    Architecture:
        [task-specific embed] → [pos embed] → [N transformer blocks] → [task-specific head]

    Transfer protocol:
        Pre-pre-training : set_task(NCA_VOCAB)   → train on NCA
        Fine-tuning      : set_task(DYCK_VOCAB)  → train on Dyck
        The transformer blocks carry their weights across.
        Only embed and head are re-initialised per task.
    """
    def __init__(self,
                 d_model : int = D_MODEL,
                 n_heads : int = N_HEADS,
                 n_layers: int = N_LAYERS,
                 d_ff    : int = D_FF,
                 max_len : int = SEQ_LEN):
        super().__init__()

        # Task-specific layers — replaced when switching tasks
        # Initialised to None; must call set_task() before forward()
        self.embed = None
        self.head  = None

        # Transferable layers — these survive the task switch
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm_out  = nn.LayerNorm(d_model)   # final norm before head

        self.d_model   = d_model
        self.max_len   = max_len

    def set_task(self, vocab_size: int):
        """
        Switch the model to a new task.
        Creates fresh embedding + head for this vocab.
        Transformer blocks are UNTOUCHED — they keep whatever weights they have.
        """
        self.embed = nn.Embedding(vocab_size, self.d_model)
        self.head  = nn.Linear(self.d_model, vocab_size, bias=False)

        # Tie embedding and output weights — reduces params, often improves performance
        # Both matrices live in the same d_model × vocab_size space
        self.head.weight = self.embed.weight

        # Initialise embedding with small normal weights
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx):
        """
        idx: (B, T) integer token indices
        returns: (B, T, vocab_size) logits
        """
        assert self.embed is not None, "Call set_task(vocab_size) before forward()"
        B, T = idx.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        # Positions [0, 1, ..., T-1] — same for every item in batch
        pos = torch.arange(T, dtype=torch.long)   # (T,)

        # Token embedding + positional embedding
        x = self.embed(idx) + self.pos_embed(pos)  # (B, T, d_model)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)                       # (B, T, d_model)
        return self.head(x)                         # (B, T, vocab_size)

    def get_transferable_state(self) -> dict:
        """
        Extract only the weights that should transfer across tasks.
        Returns a state_dict subset: blocks + pos_embed + norm_out.
        Excludes: embed, head (these are task-specific).
        """
        full_state = self.state_dict()
        transferable = {
            k: v for k, v in full_state.items()
            if k.startswith('blocks') or
               k.startswith('pos_embed') or
               k.startswith('norm_out')
        }
        return transferable

    def load_transferable_state(self, state: dict):
        """
        Load transferred weights into the model.
        strict=False means missing keys (embed, head) are silently ignored —
        they were already set by set_task().
        """
        missing, unexpected = self.load_state_dict(state, strict=False)
        # missing keys are expected (embed, head) — don't panic
        assert all('embed' in k or 'head' in k for k in missing), \
            f"Unexpected missing keys: {missing}"


def count_params(model: TinyTransformer) -> int:
    """Count trainable parameters. Should be ~150k."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

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

def make_1d_rule_network(n_states: int, hidden: int = 16, seed: int = 0) -> dict:
    """
    1D NCA rule network.
    Input: 3 values (left neighbour, self, right neighbour)
    Output: logits over n_states
    Architecture: Linear(3, hidden) → ReLU → Linear(hidden, n_states)
    """
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / 3)     # Xavier for fan_in=3
    scale2 = np.sqrt(2.0 / hidden)
    return {
        'W1': rng.normal(0, scale1, (hidden, 3)).astype(np.float32),
        'b1': np.zeros(hidden, dtype=np.float32),
        'W2': rng.normal(0, scale2, (n_states, hidden)).astype(np.float32),
        'b2': np.zeros(n_states, dtype=np.float32),
    }


def apply_1d_rule(tape: np.ndarray, rule: dict) -> np.ndarray:
    """
    Apply one 1D NCA timestep to the tape.

    tape : (L,) integer array, values in [0, n_states-1]
    Neighbourhood: left, self, right — with periodic (ring) boundaries
    """
    t    = tape.astype(np.float32)
    L    = len(t)

    # Gather 3-neighbourhood for every cell using np.roll
    # np.roll(-1) shifts left → each position sees its RIGHT neighbour
    # np.roll(+1) shifts right → each position sees its LEFT neighbour
    neighbourhood = np.stack([
        np.roll(t,  1),    # left neighbour (with wrap)
        t,                 # self
        np.roll(t, -1),    # right neighbour (with wrap)
    ], axis=-1)            # (L, 3)

    # Forward pass
    h      = np.maximum(neighbourhood @ rule['W1'].T + rule['b1'], 0)  # (L, hidden)
    logits = h @ rule['W2'].T + rule['b2']                             # (L, n_states)
    return np.argmax(logits, axis=-1)                                  # (L,)


def tokenise_1d_tape(tape: np.ndarray, n_states: int, window: int) -> list:
    """
    Convert a 1D tape into tokens using a sliding window.

    Each window of `window` consecutive cells → one token in base n_states.
    Non-overlapping windows, read left to right.

    Example (window=2, n_states=4):
      cells [2, 3] → token = 2*4 + 3 = 11
    """
    tokens = []
    for i in range(0, len(tape) - window + 1, window):   # non-overlapping
        window_vals = tape[i:i+window]
        token = 0
        for val in window_vals:
            token = token * n_states + int(val)
        tokens.append(token)
    return tokens


def generate_1d_nca_sequences(n_rules : int,
                               n_traj  : int,
                               n_steps : int,
                               n_states: int  = N_STATES,
                               tape_len: int  = TAPE_LEN,
                               window  : int  = N1D_WIN,
                               base_seed: int = 0) -> list:
    """
    Generate 1D NCA training sequences.
    Same structure as 2D version: n_rules × n_traj sequences.
    Each sequence = 2 consecutive tape steps tokenised and concatenated.
    """
    tokens_per_step = tape_len // window   # 64 // 2 = 32 tokens per step
    sequences = []
    rng = np.random.default_rng(base_seed)

    for rule_idx in range(n_rules):
        rule_seed = int(rng.integers(0, 1_000_000))
        rule = make_1d_rule_network(n_states=n_states, seed=rule_seed)

        for traj_idx in range(n_traj):
            traj_seed = int(rng.integers(0, 1_000_000))
            traj_rng  = np.random.default_rng(traj_seed)
            tape = traj_rng.integers(0, n_states, size=(tape_len,))

            step_tokens = []
            for _ in range(n_steps):
                tape = apply_1d_rule(tape, rule)
                step_tokens.append(tokenise_1d_tape(tape, n_states, window))

            # Take just the last two steps — one sequence per trajectory
            # (using all consecutive pairs gives 63x too many sequences)
            seq    = step_tokens[-2] + step_tokens[-1]   # 32 + 32 = 64 tokens
            padded = seq + [0] * (SEQ_LEN - len(seq))    # pad to SEQ_LEN=128
            sequences.append(np.array(padded[:SEQ_LEN], dtype=np.int64))

    return sequences


# Token vocabulary for Dyck-1
TOK_OPEN  = 0   # '('
TOK_CLOSE = 1   # ')'
TOK_PAD   = 2   # padding token
TOK_EOS   = 3   # end of sequence


def generate_dyck1_sequence(max_depth: int, max_len: int, rng) -> list:
    """
    Generate one valid Dyck-1 sequence (balanced parentheses).

    Strategy: random walk on a stack.
    - If stack not full: randomly open or close
    - If stack empty: must open
    - If at max_len: close everything

    Returns: list of token integers (not padded yet)
    """
    tokens  = []
    depth   = 0     # current nesting depth = number of unclosed '('

    while len(tokens) < max_len - depth:   # leave room to close all open brackets
        if depth == 0:
            tokens.append(TOK_OPEN)
            depth += 1
        elif depth >= max_depth:
            tokens.append(TOK_CLOSE)
            depth -= 1
        else:
            if rng.random() < 0.5:
                tokens.append(TOK_OPEN)
                depth += 1
            else:
                tokens.append(TOK_CLOSE)
                depth -= 1

    # Close all remaining open brackets
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


def run_condition(condition: str,
                 seed     : int,
                 nca_seqs_2d    : list,
                 nca_seqs_1d    : list,
                 nca_seqs_curriculum: list,
                 dyck_train     : torch.Tensor,
                 dyck_val       : torch.Tensor) -> list:
    """
    Run one condition × seed and return the val loss curve.

    condition: one of 'A', 'B', 'C', 'D'
      A = scratch (no pre-pre-training)
      B = 2D NCA pre-pre-training (paper's method)
      C = 1D NCA pre-pre-training (our Extension A)
      D = 2D NCA curriculum pre-pre-training (our Extension B)

    Returns: list of val losses, one per Dyck fine-tuning epoch
    """
    set_seed(seed)

    # Build model — same architecture for all conditions
    model = TinyTransformer()

    # ---- Stage 0: NCA Pre-Pre-Training (conditions B, C, D only) ----
    if condition == 'B':
        model.set_task(NCA_VOCAB)
        optimizer = torch.optim.Adam(model.parameters(), lr=NCA_LR)
        print(f"  [Seed {seed}] Condition B: 2D NCA pre-pre-training...")
        for epoch in range(NCA_EPOCHS):
            loss = train_one_epoch(model, nca_seqs_2d, optimizer)
        print(f"    Final NCA loss: {loss:.4f}")

    elif condition == 'C':
        model.set_task(NCA1D_VOCAB)
        optimizer = torch.optim.Adam(model.parameters(), lr=NCA_LR)
        print(f"  [Seed {seed}] Condition C: 1D NCA pre-pre-training...")
        for epoch in range(NCA_EPOCHS):
            loss = train_one_epoch(model, nca_seqs_1d, optimizer)
        print(f"    Final NCA loss: {loss:.4f}")

    elif condition == 'D':
        # Curriculum: 3 epochs, each over a different complexity slice
        # nca_seqs_curriculum is a list of 3 sub-lists: [simple, medium, all]
        model.set_task(NCA_VOCAB)
        optimizer = torch.optim.Adam(model.parameters(), lr=NCA_LR)
        print(f"  [Seed {seed}] Condition D: 2D NCA curriculum pre-pre-training...")
        for epoch, seqs_slice in enumerate(nca_seqs_curriculum):
            loss = train_one_epoch(model, seqs_slice, optimizer)
            print(f"    Curriculum epoch {epoch+1}: loss={loss:.4f}, "
                  f"n_seqs={len(seqs_slice)}")

    # condition A: no pre-pre-training, model weights are random init

    # ---- Stage 1: Transfer to Dyck-1 ----
    # Save transferable weights from NCA stage (if any)
    if condition != 'A':
        transferable = model.get_transferable_state()

    # Switch model to Dyck task — fresh embed + head
    model.set_task(DYCK_VOCAB)
    optimizer = torch.optim.Adam(model.parameters(), lr=DYCK_LR)

    # Load back transferable weights (blocks + pos_embed) for B/C/D
    if condition != 'A':
        model.load_transferable_state(transferable)

    # Fine-tune and track val loss
    val_losses = []
    steps_to_threshold = None
    dyck_seqs = [dyck_train[i].numpy() for i in range(len(dyck_train))]

    print(f"  [Seed {seed}] Condition {condition}: Fine-tuning on Dyck-1...")
    for epoch in range(DYCK_EPOCHS):
        train_one_epoch(model, dyck_seqs, optimizer, ignore_index=TOK_PAD)
        val_loss = evaluate(model, dyck_val)
        val_losses.append(val_loss)

        # Record first epoch we cross the convergence threshold
        if steps_to_threshold is None and val_loss < THRESHOLD:
            steps_to_threshold = epoch + 1

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d} | val_loss={val_loss:.4f}" +
                  (f" ← threshold!" if val_loss < THRESHOLD and epoch < 2 else ""))

    final_loss = val_losses[-1]
    steps_str  = str(steps_to_threshold) if steps_to_threshold else ">" + str(DYCK_EPOCHS)
    print(f"  → Final val loss: {final_loss:.4f} | Steps to threshold: {steps_str}")

    return val_losses, steps_to_threshold


def compute_gzip_complexity(sequences: list) -> np.ndarray:
    """
    Compute gzip compression ratio for each sequence.
    Lower ratio = more compressible = simpler dynamics.
    Higher ratio = less compressible = more complex dynamics.

    Returns: (N,) array of float ratios
    """
    ratios = []
    for seq in sequences:
        raw       = np.array(seq, dtype=np.uint8).tobytes()
        compressed = gzip.compress(raw)
        ratios.append(len(compressed) / len(raw))
    return np.array(ratios)


def build_curriculum(sequences: list) -> list:
    """
    Sort sequences by complexity and split into 3 curriculum stages.

    Stage 1 (epoch 1): simplest 30%  — build basic pattern tracking
    Stage 2 (epoch 2): middle 40%    — introduce moderate complexity
    Stage 3 (epoch 3): all 100%      — expose to full complexity range

    Returns: list of 3 sub-lists (one per epoch)
    """
    print("  Computing gzip complexity for curriculum sorting...")
    ratios  = compute_gzip_complexity(sequences)
    sorted_idx = np.argsort(ratios)   # ascending: simple → complex
    N = len(sorted_idx)

    simple_seqs = [sequences[i] for i in sorted_idx[:int(N * 0.30)]]
    medium_seqs = [sequences[i] for i in sorted_idx[int(N * 0.30):int(N * 0.70)]]
    all_seqs    = sequences   # all, in original (mixed) order

    print(f"  Curriculum: {len(simple_seqs)} simple | "
          f"{len(medium_seqs)} medium | {len(all_seqs)} all")

    # Epoch 1: simplest 30%
    # Epoch 2: middle 40% (new material)
    # Epoch 3: all (consolidation + hard examples)
    return [simple_seqs, medium_seqs, all_seqs] 



def main():
    t_start = time.time()
    print("=" * 60)
    print("day_1_test.py — NCA Pre-Pre-Training Experiment")
    print("=" * 60)

    # ---- 1. Generate data (done once, shared across seeds) ----
    print("\n[1/5] Generating 2D NCA sequences...")
    nca_seqs_2d = generate_2d_nca_sequences(
        n_rules=N_RULES, n_traj=N_TRAJ, n_steps=N_STEPS, base_seed=0
    )
    print(f"  Generated {len(nca_seqs_2d):,} 2D NCA sequences")

    print("[2/5] Generating 1D NCA sequences...")
    nca_seqs_1d = generate_1d_nca_sequences(
        n_rules=N_RULES, n_traj=N_TRAJ, n_steps=N_STEPS, base_seed=1
    )
    print(f"  Generated {len(nca_seqs_1d):,} 1D NCA sequences")

    print("[3/5] Building complexity curriculum for Condition D...")
    nca_seqs_curriculum = build_curriculum(nca_seqs_2d)

    print("[4/5] Generating Dyck-1 dataset...")
    dyck_train = make_dyck_dataset(N_DYCK_TRAIN, seed=100)
    dyck_val   = make_dyck_dataset(N_DYCK_VAL,   seed=101)
    print(f"  Train: {len(dyck_train):,} | Val: {len(dyck_val):,}")

    # ---- 2. Run all conditions × seeds ----
    print("\n[5/5] Running experiments...")
    conditions = ['A', 'B', 'C', 'D']
    condition_labels = {
        'A': 'Scratch',
        'B': '2D NCA (paper)',
        'C': '1D NCA (ours)',
        'D': '2D NCA + Curriculum (ours)',
    }

    # results[condition] = list of (val_losses, steps_to_threshold) per seed
    results    = defaultdict(list)
    all_curves = defaultdict(list)   # for plotting

    for condition in conditions:
        print(f"\n--- Condition {condition}: {condition_labels[condition]} ---")
        for seed in SEEDS:
            val_losses, steps = run_condition(
                condition, seed,
                nca_seqs_2d, nca_seqs_1d, nca_seqs_curriculum,
                dyck_train, dyck_val
            )
            results[condition].append((val_losses[-1], steps))
            all_curves[condition].append(val_losses)

    # ---- 3. Print summary table ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<30} {'Final Val Loss':>15} {'Steps to <0.3':>15}")
    print("-" * 60)

    for condition in conditions:
        final_losses = [r[0] for r in results[condition]]
        steps_list   = [r[1] for r in results[condition] if r[1] is not None]

        mean_loss = np.mean(final_losses)
        std_loss  = np.std(final_losses)
        mean_steps = np.mean(steps_list) if steps_list else float('nan')

        label = f"{condition}: {condition_labels[condition]}"
        steps_str = f"{mean_steps:.0f}" if not np.isnan(mean_steps) else ">20"
        print(f"{label:<30} {mean_loss:.4f} ± {std_loss:.4f}   {steps_str:>10}")

    # Transfer efficiency vs condition A
    scratch_steps = np.mean([r[1] for r in results['A'] if r[1] is not None])
    print("\nTransfer Efficiency (vs Scratch):")
    for cond in ['B', 'C', 'D']:
        cond_steps = [r[1] for r in results[cond] if r[1] is not None]
        if cond_steps and not np.isnan(scratch_steps):
            efficiency = scratch_steps / np.mean(cond_steps)
            print(f"  Condition {cond}: {efficiency:.2f}x faster convergence")

    # ---- 4. Save loss curves plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = {'A': '#888888', 'B': '#4dabf7', 'C': '#69db7c', 'D': '#ffa94d'}
    epochs  = list(range(1, DYCK_EPOCHS + 1))

    for condition in conditions:
        curves = np.array(all_curves[condition])   # (n_seeds, n_epochs)
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        ax.plot(epochs, mean, color=colours[condition],
                label=f"{condition}: {condition_labels[condition]}",
                linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=colours[condition], alpha=0.2)

    ax.axhline(THRESHOLD, color='white', linestyle='--', alpha=0.5,
               label=f'Convergence threshold ({THRESHOLD})')
    ax.set_xlabel('Fine-tuning epoch', fontsize=12)
    ax.set_ylabel('Validation loss', fontsize=12)
    ax.set_title('Dyck-1 Fine-tuning: NCA Pre-Pre-Training vs Scratch\n'
                 'shaded = ± 1 std across 3 seeds', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nLoss curves saved to: loss_curves.png")

    elapsed = time.time() - t_start
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == '__main__':
    main()


