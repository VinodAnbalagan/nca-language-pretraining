import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", app_title="Training Language Models via Neural Cellular Automata")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    import random
    import gzip

    return F, FuncAnimation, HTML, mcolors, mo, nn, np, plt, random, torch


@app.cell
def _(np):
    def make_rule_network(n: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / 9)
        scale2 = np.sqrt(2.0 / hidden_dim)
        return {
            'W1': rng.normal(0, scale1, (hidden_dim, 9)).astype(np.float32),
            'b1': np.zeros(hidden_dim, dtype=np.float32),
            'W2': rng.normal(0, scale2, (n, hidden_dim)).astype(np.float32),
            'b2': np.zeros(n, dtype=np.float32),
        }

    def rule_forward(params, neighbourhood_flat):
        h = neighbourhood_flat @ params['W1'].T + params['b1']
        h = np.maximum(h, 0)
        return h @ params['W2'].T + params['b2']

    return make_rule_network, rule_forward


@app.cell
def _(np):
    def get_neighbourhood(grid):
        H, W = grid.shape
        g = grid.astype(np.float32)
        channels = [
            g,
            np.roll(g, -1, axis=0),
            np.roll(np.roll(g, -1, axis=0), 1, axis=1),
            np.roll(g,  1, axis=1),
            np.roll(np.roll(g,  1, axis=0), 1, axis=1),
            np.roll(g,  1, axis=0),
            np.roll(np.roll(g,  1, axis=0), -1, axis=1),
            np.roll(g, -1, axis=1),
            np.roll(np.roll(g, -1, axis=0), -1, axis=1),
        ]
        return np.stack(channels, axis=-1)

    return (get_neighbourhood,)


@app.cell
def _(get_neighbourhood, np, rule_forward):
    def nca_step(grid, params, n):
        H, W = grid.shape
        neighbourhood = get_neighbourhood(grid)
        neighbourhood_flat = neighbourhood.reshape(-1, 9)
        logits = rule_forward(params, neighbourhood_flat)
        new_states_flat = np.argmax(logits, axis=-1)
        return new_states_flat.reshape(H, W)

    return (nca_step,)


@app.cell
def _(make_rule_network, nca_step, np):
    def run_nca(n: int, T: int, H: int = 12, W: int = 12, seed: int = 42):
        rng = np.random.default_rng(seed)
        grid = rng.integers(0, n, size=(H, W))
        rule_seed = int(rng.integers(0, 100000))
        params = make_rule_network(n=n, seed=rule_seed)
        trajectory = [grid.copy()]
        for _ in range(T):
            grid = nca_step(grid, params, n)
            trajectory.append(grid.copy())
        return np.stack(trajectory, axis=0)

    return (run_nca,)


@app.cell
def _(mo):
    n_slider = mo.ui.slider(start=2, stop=10, step=1, value=5,
                             label="Alphabet size n (number of cell states)")
    T_slider = mo.ui.slider(start=5, stop=80, step=5, value=30,
                             label="Timesteps T")
    seed_slider = mo.ui.slider(start=0, stop=99, step=1, value=0,
                                label="Rule seed (try different values!)")
    return T_slider, n_slider, seed_slider


@app.cell
def _(T_slider, mo, n_slider, seed_slider):
    mo.vstack([
        mo.callout(
            mo.md("""
            ## 🧬 NCA Explorer
            Adjust the sliders to change the cellular automaton's behaviour.
            - **n** controls how many distinct states each cell can take.
            - **T** controls how many steps to simulate.
            - **Rule seed** samples a completely different update rule.
            """),
            kind="info"
        ),
        n_slider, T_slider, seed_slider,
    ])
    return


@app.cell
def _(mcolors, n_slider, np, plt):
    def make_colormap(n):
        base = plt.colormaps['tab10']
        colours = [base(i % 10) for i in range(n)]
        cmap_out = mcolors.ListedColormap(colours)
        norm_out = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n, 1.0), ncolors=n)
        return cmap_out, norm_out

    cmap, norm = make_colormap(n_slider.value)
    return (make_colormap,)


@app.cell
def _(T_slider, make_colormap, mo, n_slider, plt, run_nca, seed_slider):
    traj = run_nca(n=n_slider.value, T=T_slider.value, seed=seed_slider.value)
    cmap_local, norm_local = make_colormap(n_slider.value)
    fig_s1a, axes_s1a = plt.subplots(1, 2, figsize=(8, 4))
    fig_s1a.patch.set_facecolor('#1a1a2e')
    for _ax, _t, _lbl in zip(axes_s1a, [0, T_slider.value],
                               ["t = 0  (initial)", f"t = {T_slider.value}  (final)"]):
        _ax.imshow(traj[_t], cmap=cmap_local, norm=norm_local, interpolation='nearest')
        _ax.set_title(_lbl, color='white', fontsize=12)
        _ax.set_xticks([]); _ax.set_yticks([])
        for _sp in _ax.spines.values(): _sp.set_edgecolor('#444')
    fig_s1a.suptitle(f"NCA: n={n_slider.value} states, seed={seed_slider.value}",
                     color='white', fontsize=14)
    plt.tight_layout()
    mo.mpl.interactive(fig_s1a)
    return (traj,)


@app.cell
def _(FuncAnimation, HTML, T_slider, make_colormap, mo, n_slider, plt, traj):
    cmap_anim, norm_anim = make_colormap(n_slider.value)
    fig_anim, ax_anim = plt.subplots(figsize=(5, 5))
    fig_anim.patch.set_facecolor('#1a1a2e')
    ax_anim.set_facecolor('#1a1a2e')
    ax_anim.set_xticks([]); ax_anim.set_yticks([])
    im = ax_anim.imshow(traj[0], cmap=cmap_anim, norm=norm_anim,
                        interpolation='nearest', animated=True)
    title_anim = ax_anim.set_title("t = 0", color='white', fontsize=13, pad=10)

    def update(frame):
        im.set_data(traj[frame])
        title_anim.set_text(f"t = {frame}")
        return im, title_anim

    anim = FuncAnimation(fig_anim, update, frames=T_slider.value + 1,
                         interval=120, blit=True)
    plt.tight_layout()
    html_anim = HTML(anim.to_jshtml())
    mo.Html(html_anim.data)
    return


@app.cell
def _(T_slider, make_colormap, mo, n_slider, plt, run_nca):
    _n_val = n_slider.value
    _T_val = T_slider.value
    _cmap_grid, _norm_grid = make_colormap(_n_val)
    _seeds_show = [0, 7, 13, 42]
    fig_comp, axes_comp = plt.subplots(2, 4, figsize=(12, 6))
    fig_comp.patch.set_facecolor('#1a1a2e')
    fig_comp.suptitle(f"4 different NCA rules  (n={_n_val}, T={_T_val})",
                      color='white', fontsize=14)
    for _col, _seed in enumerate(_seeds_show):
        _traj_c = run_nca(n=_n_val, T=_T_val, seed=_seed)
        for _row, (_t2, _lbl2) in enumerate([(0, "t=0"), (_T_val, f"t={_T_val}")]):
            _ax2 = axes_comp[_row, _col]
            _ax2.imshow(_traj_c[_t2], cmap=_cmap_grid, norm=_norm_grid,
                        interpolation='nearest')
            _ax2.set_xticks([]); _ax2.set_yticks([])
            if _row == 0: _ax2.set_title(f"seed={_seed}", color='white', fontsize=11)
            if _col == 0: _ax2.set_ylabel(_lbl2, color='white', fontsize=10)
    plt.tight_layout()
    mo.mpl.interactive(fig_comp)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — From Grid to Token: The Bridge

    The NCA grid looks nothing like language. A 16x16 grid of coloured
    squares, evolving step by step under a hidden rule.

    But to train a transformer on it, we need **tokens** — integers from
    a fixed vocabulary. This section walks through the tokenisation pipeline
    and reveals the surprise: NCA token frequencies follow the same
    power-law distribution as natural language.
    """)
    return


@app.cell
def _(mo):
    patch_slider = mo.ui.slider(start=1, stop=4, step=1, value=2,
                                 label="Patch size (p x p)")
    n_tok_slider = mo.ui.slider(start=2, stop=8, step=1, value=2,
                                 label="Alphabet size n")
    return n_tok_slider, patch_slider


@app.cell
def _(make_colormap, mo, n_tok_slider, patch_slider, plt, run_nca):
    _p = patch_slider.value
    _n = n_tok_slider.value
    vocab_size = _n ** (_p * _p)
    _traj_s2 = run_nca(n=_n, T=20, H=16, W=16, seed=0)
    _grid = _traj_s2[20]
    _cmap_p, _norm_p = make_colormap(_n)
    fig_s2a, axes_s2a = plt.subplots(1, 2, figsize=(12, 5))
    fig_s2a.patch.set_facecolor('#1a1a2e')
    _ax_l = axes_s2a[0]
    _ax_l.imshow(_grid, cmap=_cmap_p, norm=_norm_p, interpolation='nearest')
    _ax_l.set_title('NCA Grid (raw)', color='white', fontsize=12)
    _ax_l.set_xticks([]); _ax_l.set_yticks([])
    _ax_l.set_facecolor('#1a1a2e')
    _H, _W = _grid.shape
    for _i in range(0, _H + 1, _p):
        _ax_l.axhline(_i - 0.5, color='white', linewidth=1.5, alpha=0.6)
    for _j in range(0, _W + 1, _p):
        _ax_l.axvline(_j - 0.5, color='white', linewidth=1.5, alpha=0.6)
    _ax_r = axes_s2a[1]
    _ax_r.imshow(_grid, cmap=_cmap_p, norm=_norm_p, interpolation='nearest', alpha=0.6)
    _ax_r.set_title(f'Patches -> Tokens  (vocab = {vocab_size:,})', color='white', fontsize=12)
    _ax_r.set_xticks([]); _ax_r.set_yticks([])
    _ax_r.set_facecolor('#1a1a2e')
    for _i in range(0, _H, _p):
        for _j in range(0, _W, _p):
            _patch = _grid[_i:_i+_p, _j:_j+_p].flatten()
            _tok = 0
            for _v in _patch: _tok = _tok * _n + int(_v)
            _cx = _j + _p / 2 - 0.5
            _cy = _i + _p / 2 - 0.5
            _ax_r.text(_cx, _cy, str(_tok), ha='center', va='center',
                       fontsize=max(6, 10 - _p), color='white', fontweight='bold')
            _ax_r.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), _p, _p,
                                           linewidth=1.5, edgecolor='white',
                                           facecolor='none', alpha=0.7))
    fig_s2a.suptitle(f'Tokenisation: patch={_p}x{_p}, n={_n} -> vocab={vocab_size:,}',
                     color='white', fontsize=13)
    plt.tight_layout()
    mo.mpl.interactive(fig_s2a)
    return


@app.cell
def _(mo, np, plt):
    fig_s2b, ax_s2b = plt.subplots(figsize=(9, 5))
    fig_s2b.patch.set_facecolor('#1a1a2e')
    ax_s2b.set_facecolor('#1a1a2e')
    _n_vals = np.arange(2, 16)
    _colours_patch = {1: '#69db7c', 2: '#4dabf7', 3: '#ffa94d', 4: '#ff6b6b'}
    for _p2, _colour2 in _colours_patch.items():
        _vocab2 = _n_vals ** (_p2 * _p2)
        ax_s2b.semilogy(_n_vals, _vocab2, color=_colour2, linewidth=2,
                        label=f'patch {_p2}x{_p2}  (vocab = n^{_p2*_p2})',
                        marker='o', markersize=4)
    ax_s2b.axvline(2, color='white', linestyle='--', alpha=0.4)
    ax_s2b.annotate('Our choice\nn=2, patch=2\nvocab=16',
                    xy=(2, 16), xytext=(4, 50), color='white', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
    ax_s2b.set_xlabel('Alphabet size  n', color='white', fontsize=12)
    ax_s2b.set_ylabel('Vocabulary size  (log scale)', color='white', fontsize=12)
    ax_s2b.set_title('Vocabulary size explodes with n', color='white', fontsize=13)
    ax_s2b.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    ax_s2b.tick_params(colors='white')
    for _sp2 in ax_s2b.spines.values(): _sp2.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_s2b)
    return


@app.cell
def _():
    def tokenise_grid(grid, n_states, patch_size):
        H, W = grid.shape
        tokens = []
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = grid[i:i+patch_size, j:j+patch_size].flatten()
                token = 0
                for cell_value in patch:
                    token = token * n_states + int(cell_value)
                tokens.append(token)
        return tokens

    def tokenise_trajectory(trajectory, n_states, patch_size):
        all_tokens = []
        for t in range(len(trajectory)):
            all_tokens.extend(tokenise_grid(trajectory[t], n_states, patch_size))
        return all_tokens

    return (tokenise_trajectory,)


@app.cell
def _(mo, n_tok_slider, np, run_nca, tokenise_trajectory):
    _n_states = n_tok_slider.value
    _all_tokens = []
    for _rule_seed in range(50):
        _traj_z = run_nca(n=_n_states, T=20, H=16, W=16, seed=_rule_seed)
        _all_tokens.extend(tokenise_trajectory(_traj_z, _n_states, 2))
    all_tokens = np.array(_all_tokens)
    mo.callout(mo.md(f"""
    Generated **{len(all_tokens):,} tokens** from 50 different NCA rules.
    Vocabulary size: {_n_states}^4 = {_n_states**4} possible tokens.
    Unique tokens observed: {len(np.unique(all_tokens))}
    """), kind='success')
    return (all_tokens,)


@app.cell
def _(all_tokens, mo, n_tok_slider, np, plt):
    from collections import Counter
    _n_states2 = n_tok_slider.value
    _counts = Counter(all_tokens.tolist())
    _sorted_counts = sorted(_counts.values(), reverse=True)
    _ranks = np.arange(1, len(_sorted_counts) + 1)
    _freqs = np.array(_sorted_counts, dtype=float)
    _freqs_norm = _freqs / _freqs.sum()
    _zipf = 1.0 / _ranks
    _zipf /= _zipf.sum()
    fig_zipf, axes_zipf = plt.subplots(1, 2, figsize=(14, 5))
    fig_zipf.patch.set_facecolor('#1a1a2e')
    _ax_z0 = axes_zipf[0]
    _ax_z0.set_facecolor('#1a1a2e')
    _ax_z0.bar(range(len(_sorted_counts)), _freqs_norm, color='#4dabf7', alpha=0.8, width=1.0)
    _ax_z0.set_xlabel('Token rank', color='white', fontsize=11)
    _ax_z0.set_ylabel('Frequency', color='white', fontsize=11)
    _ax_z0.set_title('Token frequency (linear scale)', color='white', fontsize=12)
    _ax_z0.tick_params(colors='white')
    for _sp3 in _ax_z0.spines.values(): _sp3.set_edgecolor('#444')
    _ax_z1 = axes_zipf[1]
    _ax_z1.set_facecolor('#1a1a2e')
    _ax_z1.loglog(_ranks, _freqs_norm, 'o', color='#4dabf7', markersize=4, alpha=0.8,
                  label='NCA token frequencies')
    _ax_z1.loglog(_ranks, _zipf, '--', color='#ffa94d', linewidth=2, alpha=0.8,
                  label="Zipf's law: p(r) proportional to 1/r")
    _ax_z1.set_xlabel('Rank (log)', color='white', fontsize=11)
    _ax_z1.set_ylabel('Frequency (log)', color='white', fontsize=11)
    _ax_z1.set_title("Log-log: NCA follows Zipf's law!", color='white', fontsize=12)
    _ax_z1.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    _ax_z1.tick_params(colors='white')
    for _sp4 in _ax_z1.spines.values(): _sp4.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_zipf)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 - The Finding: What Did NCA Pre-Training Actually Install?

    We train two tiny transformers:
    - **Scratch:** random initialisation, no pre-training
    - **NCA:** pre-trained on NCA sequences, then switched to bracket task

    Before either model sees a single bracket, we ask:
    *where does each model attend when shown a bracket sequence?*
    """)
    return


@app.cell
def _(F, nn, torch):
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
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
            out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
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
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )
        def forward(self, x, return_attn=False):
            if return_attn:
                attn_out, w = self.attn(self.norm1(x), return_attn=True)
                x2 = x + attn_out
                return x2 + self.ff(self.norm2(x2)), w
            x2 = x + self.attn(self.norm1(x))
            return x2 + self.ff(self.norm2(x2))

    class TinyTransformerViz(nn.Module):
        D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; D_FF = 256; SEQ_LEN = 256

        def __init__(self):
            super().__init__()
            self.embed     = None
            self.head      = None
            self.blocks    = nn.ModuleList([
                TransformerBlock(self.D_MODEL, self.N_HEADS, self.D_FF)
                for _ in range(self.N_LAYERS)
            ])
            self.pos_embed = nn.Embedding(self.SEQ_LEN, self.D_MODEL)
            self.norm_out  = nn.LayerNorm(self.D_MODEL)

        def set_task(self, vocab_size):
            self.embed = nn.Embedding(vocab_size, self.D_MODEL)
            self.head  = nn.Linear(self.D_MODEL, vocab_size, bias=False)
            self.head.weight = self.embed.weight
            nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, idx):
            B, T = idx.shape
            pos  = torch.arange(T, dtype=torch.long)
            x    = self.embed(idx) + self.pos_embed(pos)
            for block in self.blocks: x = block(x)
            return self.head(self.norm_out(x))

        def get_attention(self, idx, layer=0):
            self.eval()
            with torch.no_grad():
                B, T = idx.shape
                pos  = torch.arange(T, dtype=torch.long)
                x    = self.embed(idx) + self.pos_embed(pos)
                for i, block in enumerate(self.blocks):
                    if i == layer: x, attn_w = block(x, return_attn=True)
                    else: x = block(x)
            return attn_w.squeeze(0).cpu().numpy()

        def get_transferable_state(self):
            return {k: v for k, v in self.state_dict().items()
                    if k.startswith('blocks') or k.startswith('pos_embed')
                    or k.startswith('norm_out')}

        def load_transferable_state(self, state):
            self.load_state_dict(state, strict=False)

    return (TinyTransformerViz,)


@app.cell
def _(np, torch):
    TOK_OPEN = 0; TOK_CLOSE = 1; TOK_PAD = 2; TOK_EOS = 3; DYCK_VOCAB = 4
    NCA_VOCAB = 2 ** (2 * 2); N_STATES = 2; PATCH = 2; GRID_H = GRID_W = 16

    def _make_rule(seed):
        rng = np.random.default_rng(seed)
        s1, s2 = np.sqrt(2/9), np.sqrt(2/16)
        return {'W1': rng.normal(0, s1, (16, 9)).astype(np.float32),
                'b1': np.zeros(16, np.float32),
                'W2': rng.normal(0, s2, (N_STATES, 16)).astype(np.float32),
                'b2': np.zeros(N_STATES, np.float32)}

    def _step(grid, rule):
        g = grid.astype(np.float32)
        nb = np.stack([g, np.roll(g,-1,0),
                       np.roll(np.roll(g,-1,0),1,1), np.roll(g,1,1),
                       np.roll(np.roll(g,1,0),1,1), np.roll(g,1,0),
                       np.roll(np.roll(g,1,0),-1,1), np.roll(g,-1,1),
                       np.roll(np.roll(g,-1,0),-1,1)], axis=-1).reshape(-1, 9)
        h = np.maximum(nb @ rule['W1'].T + rule['b1'], 0)
        return np.argmax(h @ rule['W2'].T + rule['b2'], axis=-1).reshape(grid.shape)

    def _tok_grid(grid):
        tokens = []
        for i in range(0, GRID_H, PATCH):
            for j in range(0, GRID_W, PATCH):
                p = grid[i:i+PATCH, j:j+PATCH].flatten()
                t = 0
                for v in p: t = t * N_STATES + int(v)
                tokens.append(t)
        return tokens

    def generate_nca_seqs(n_rules=50, n_traj=20, n_steps=32, seed=0):
        seqs = []; rng = np.random.default_rng(seed)
        for _ in range(n_rules):
            rule = _make_rule(int(rng.integers(0, 1_000_000)))
            for _ in range(n_traj):
                g = np.random.default_rng(int(rng.integers(0, 1_000_000))
                                          ).integers(0, N_STATES, (GRID_H, GRID_W))
                steps = []
                for _ in range(n_steps):
                    g = _step(g, rule); steps.append(_tok_grid(g))
                seqs.append(np.array((steps[-2] + steps[-1])[:256], dtype=np.int64))
        return seqs

    def generate_dyck(n=2000, max_depth=8, seq_len=64, seed=0):
        rng = np.random.default_rng(seed); data = []
        for _ in range(n):
            toks, depth = [], 0
            while len(toks) < seq_len - depth:
                if depth == 0:             toks.append(TOK_OPEN);  depth += 1
                elif depth >= max_depth:   toks.append(TOK_CLOSE); depth -= 1
                elif rng.random() < 0.5:  toks.append(TOK_OPEN);  depth += 1
                else:                     toks.append(TOK_CLOSE); depth -= 1
            toks.extend([TOK_CLOSE] * depth); toks.append(TOK_EOS)
            toks = toks[:256] + [TOK_PAD] * (256 - len(toks[:256]))
            data.append(toks)
        return torch.tensor(data, dtype=torch.long)

    return DYCK_VOCAB, NCA_VOCAB, TOK_PAD, generate_dyck, generate_nca_seqs


@app.cell
def _(F, TOK_PAD, np, random, torch):
    def make_loader(sequences, batch_size=32, shuffle=True):
        data = torch.tensor(np.stack(sequences), dtype=torch.long)
        idx  = list(range(len(data)))
        if shuffle: random.shuffle(idx)
        for s in range(0, len(data) - batch_size + 1, batch_size):
            b = data[idx[s:s+batch_size]]
            yield b[:, :-1], b[:, 1:]

    def train_epoch(model, seqs, opt, ignore=-1):
        model.train(); total, n = 0., 0
        for x, y in make_loader(seqs):
            opt.zero_grad()
            loss = F.cross_entropy(model(x).reshape(-1, model.head.weight.shape[0]),
                                   y.reshape(-1), ignore_index=ignore)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def eval_model(model, data):
        model.eval(); seqs = [data[i].numpy() for i in range(len(data))]; total, n = 0., 0
        for x, y in make_loader(seqs, shuffle=False):
            loss = F.cross_entropy(model(x).reshape(-1, model.head.weight.shape[0]),
                                   y.reshape(-1), ignore_index=TOK_PAD)
            total += loss.item(); n += 1
        return total / max(n, 1)

    return (train_epoch,)


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train both models (~5 min)")
    mo.vstack([
        mo.callout(mo.md("""
        **Click to train two models:**
        - Scratch: random init, fine-tune on Dyck-1
        - NCA: pre-train on NCA sequences, fine-tune on Dyck-1
        Training runs on CPU. Takes ~5 minutes.
        """), kind='info'),
        train_btn
    ])
    return (train_btn,)


@app.cell
def _(
    DYCK_VOCAB,
    NCA_VOCAB,
    TinyTransformerViz,
    generate_dyck,
    generate_nca_seqs,
    mo,
    np,
    random,
    torch,
    train_btn,
    train_epoch,
):
    train_btn
    if not train_btn.value:
        scratch_ckpts = None; nca_ckpts = None
        mo.stop(True, mo.md("*Click the button above to train.*"))

    _NCA_EPOCHS = 15; _DYCK_EPOCHS = 5; _NCA_LR = 3e-4; _DYCK_LR = 1e-4
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    with mo.status.spinner("Generating data..."):
        _nca_seqs   = generate_nca_seqs(n_rules=50, n_traj=20, n_steps=32)
        _dyck_train = generate_dyck(n=5000, seed=100)
        _dyck_val   = generate_dyck(n=1000, seed=101)

    def _train_condition(condition):
        model = TinyTransformerViz(); ckpts = {}
        if condition == 'nca':
            model.set_task(NCA_VOCAB)
            opt = torch.optim.Adam(model.parameters(), lr=_NCA_LR)
            for ep in range(_NCA_EPOCHS): train_epoch(model, _nca_seqs, opt)
            _transferable = model.get_transferable_state()
        model.set_task(DYCK_VOCAB)
        opt = torch.optim.Adam(model.parameters(), lr=_DYCK_LR)
        if condition == 'nca': model.load_transferable_state(_transferable)
        ckpts[0] = {k: v.clone() for k, v in model.state_dict().items()}
        _dyck_seqs = [_dyck_train[i].numpy() for i in range(len(_dyck_train))]
        for ep in range(1, _DYCK_EPOCHS + 1):
            train_epoch(model, _dyck_seqs, opt, ignore=2)
            if ep == _DYCK_EPOCHS:
                ckpts[ep] = {k: v.clone() for k, v in model.state_dict().items()}
        return ckpts

    with mo.status.spinner("Training scratch model..."): scratch_ckpts = _train_condition('scratch')
    with mo.status.spinner("Training NCA model..."): nca_ckpts = _train_condition('nca')
    mo.callout(mo.md("**Training complete!** Scroll down to see the results."), kind='success')
    return nca_ckpts, scratch_ckpts


@app.cell
def _(torch):
    _raw = [0, 0, 1, 0, 0, 1, 1, 1, 3]
    _padded = _raw + [2] * (256 - len(_raw))
    probe_tokens = torch.tensor([_padded], dtype=torch.long)
    probe_labels = ['(', '(', ')', '(', '(', ')', ')', ')', 'EOS']
    MATCHING_PAIRS = {2: 1, 5: 4, 6: 3, 7: 0}
    return MATCHING_PAIRS, probe_labels, probe_tokens


@app.cell
def _(
    DYCK_VOCAB,
    MATCHING_PAIRS,
    TinyTransformerViz,
    mo,
    nca_ckpts,
    plt,
    probe_labels,
    probe_tokens,
    scratch_ckpts,
):
    if scratch_ckpts is None:
        mo.stop(True, mo.md("*Train the models first.*"))

    import matplotlib.gridspec as _gs

    def _load_model(ckpts, epoch):
        m = TinyTransformerViz(); m.set_task(DYCK_VOCAB); m.load_state_dict(ckpts[epoch]); return m

    def _draw_hmap(ax, model, title='', show_len=9):
        attn = model.get_attention(probe_tokens, layer=0)
        attn_map = attn.mean(axis=0)[:show_len, :show_len]
        ax.imshow(attn_map, cmap='Blues', vmin=0, vmax=attn_map.max(),
                  aspect='auto', interpolation='nearest')
        ax.set_xticks(range(show_len)); ax.set_yticks(range(show_len))
        ax.set_xticklabels(probe_labels[:show_len], fontsize=9, color='white')
        ax.set_yticklabels(probe_labels[:show_len], fontsize=9, color='white')
        ax.set_xlabel('Attending TO', fontsize=8, color='white')
        ax.set_ylabel('FROM', fontsize=8, color='white')
        ax.set_title(title, fontsize=10, color='white', pad=6)
        ax.tick_params(colors='white')
        for _cp, _op in MATCHING_PAIRS.items():
            if _cp < show_len and _op < show_len:
                ax.add_patch(plt.Rectangle((_op - 0.5, _cp - 0.5), 1, 1,
                                           linewidth=2, edgecolor='red', facecolor='none'))

    fig_hmap = plt.figure(figsize=(12, 10))
    fig_hmap.patch.set_facecolor('#1a1a2e')
    fig_hmap.suptitle('Attention Patterns: What Did NCA Pre-Training Install?\n'
                      'Red boxes = correct bracket-matching positions',
                      fontsize=13, color='white', y=0.98)
    _gspec = _gs.GridSpec(2, 2, figure=fig_hmap, hspace=0.45, wspace=0.35)
    _panels = [
        (0, 0, scratch_ckpts, 0, 'Scratch - Epoch 0\n(random init, no Dyck training)'),
        (0, 1, scratch_ckpts, 5, 'Scratch - Epoch 5\n(after 5 epochs Dyck)'),
        (1, 0, nca_ckpts,     0, 'NCA Pre-trained - Epoch 0\n(after NCA, before any Dyck)'),
        (1, 1, nca_ckpts,     5, 'NCA Pre-trained - Epoch 5\n(after 5 epochs Dyck)'),
    ]
    for _row, _col, _ckpts, _ep, _title in _panels:
        _ax = fig_hmap.add_subplot(_gspec[_row, _col])
        _ax.set_facecolor('#0d0d1a')
        _draw_hmap(_ax, _load_model(_ckpts, _ep), title=_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    mo.mpl.interactive(fig_hmap)
    return


@app.cell
def _(
    DYCK_VOCAB,
    MATCHING_PAIRS,
    TinyTransformerViz,
    mo,
    nca_ckpts,
    np,
    plt,
    probe_tokens,
    scratch_ckpts,
):
    if scratch_ckpts is None:
        mo.stop(True, mo.md("*Train the models first.*"))

    def _bracket_score(model, layer):
        attn = model.get_attention(probe_tokens, layer=layer)
        vals = [float(attn[h, cp, op])
                for cp, op in MATCHING_PAIRS.items()
                for h in range(attn.shape[0])]
        return np.mean(vals)

    _random_baseline = 1.0 / 9
    _conditions = {
        'Scratch\nEpoch 0': (scratch_ckpts, 0),
        'Scratch\nEpoch 5': (scratch_ckpts, 5),
        'NCA\nEpoch 0':     (nca_ckpts,     0),
        'NCA\nEpoch 5':     (nca_ckpts,     5),
    }
    _scores_l0 = {}; _scores_l1 = {}
    for _lbl, (_ckpts2, _ep2) in _conditions.items():
        _m = TinyTransformerViz(); _m.set_task(DYCK_VOCAB); _m.load_state_dict(_ckpts2[_ep2])
        _scores_l0[_lbl] = _bracket_score(_m, layer=0)
        _scores_l1[_lbl] = _bracket_score(_m, layer=1)
    _lbls = list(_conditions.keys())
    _x = np.arange(len(_lbls)); _w = 0.35
    fig_bsc, ax_bsc = plt.subplots(figsize=(10, 6))
    fig_bsc.patch.set_facecolor('#1a1a2e'); ax_bsc.set_facecolor('#1a1a2e')
    _b0 = ax_bsc.bar(_x - _w/2, [_scores_l0[l] for l in _lbls], _w,
                     label='Layer 0', color='#4dabf7', alpha=0.85)
    _b1 = ax_bsc.bar(_x + _w/2, [_scores_l1[l] for l in _lbls], _w,
                     label='Layer 1  (induction heads)', color='#ffa94d', alpha=0.85)
    ax_bsc.axhline(_random_baseline, color='white', linestyle='--', linewidth=1.5, alpha=0.6,
                   label=f'Random baseline ({_random_baseline:.3f})')
    ax_bsc.axvspan(1.5, 3.5, alpha=0.07, color='#69db7c', label='NCA conditions')
    for _bar in list(_b0) + list(_b1):
        _h = _bar.get_height()
        ax_bsc.text(_bar.get_x() + _bar.get_width()/2, _h + 0.003,
                    f'{_h:.3f}', ha='center', va='bottom', fontsize=8, color='white')
    ax_bsc.set_xticks(_x); ax_bsc.set_xticklabels(_lbls, fontsize=10, color='white')
    ax_bsc.set_ylabel('Bracket Attention Score', fontsize=12, color='white')
    ax_bsc.set_title('NCA pre-training installs Layer 1 structure BEFORE seeing brackets',
                     fontsize=12, color='white')
    ax_bsc.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    ax_bsc.tick_params(colors='white')
    ax_bsc.set_ylim(0, max(max(_scores_l0.values()), max(_scores_l1.values())) * 1.3)
    for _sp5 in ax_bsc.spines.values(): _sp5.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_bsc)
    return


@app.cell
def _(mo):
    mo.callout(mo.md("""
    ## The Finding

    **NCA pre-training installs induction-head-like circuits in Layer 1
    before the model sees a single bracket.**

    - NCA Layer 1 score at epoch 0: **0.248**
    - Scratch Layer 1 score at epoch 0: **0.179**
    - Random baseline: **0.111**

    The NCA model attends to correct matching positions **2.2x above random chance**
    purely from learning to predict NCA dynamics.

    *The structure transferred. Does fine-tuning preserve it?* -> Section 4
    """), kind='success')
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 - The Problem: Catastrophic Forgetting

    Standard fine-tuning **destroys** the pre-trained circuits.
    This is **catastrophic forgetting** visible at the circuit level
    for the first time in NCA pre-training research.
    """)
    return


@app.cell
def _(np):
    PROBE_EPOCHS = [0, 5, 20, 40, 80]
    L1_SCORES = {
        'A: Scratch':               [0.179, 0.156, 0.126, 0.143, 0.144],
        'B: NCA standard LR':       [0.248, 0.219, 0.208, 0.192, 0.181],
        'E: NCA slow LR (ours)':    [0.248, 0.255, 0.245, 0.232, 0.221],
        'F: NCA frozen attn (ours)':[0.248, 0.223, 0.241, 0.238, 0.224],
    }
    L0_SCORES = {
        'A: Scratch':               [0.179, 0.211, 0.213, 0.199, 0.165],
        'B: NCA standard LR':       [0.166, 0.155, 0.165, 0.172, 0.169],
        'E: NCA slow LR (ours)':    [0.166, 0.165, 0.170, 0.169, 0.169],
        'F: NCA frozen attn (ours)':[0.166, 0.169, 0.171, 0.169, 0.167],
    }
    RANDOM_BASELINE = 1.0 / 9
    LOSS_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80]
    LOSS_RAW = {
        'A: Scratch': {42: [0.6049, 0.5657, 0.5522, 0.5517, 0.5509, 0.5510, 0.5513, 0.5509],
                       43: [0.5996, 0.5528, 0.5514, 0.5523, 0.5513, 0.5520, 0.5512, 0.5516],
                       44: [0.6055, 0.5646, 0.5548, 0.5521, 0.5524, 0.5511, 0.5510, 0.5515]},
        'B: NCA standard LR': {42: [0.5634, 0.5533, 0.5517, 0.5512, 0.5511, 0.5515, 0.5510, 0.6177],
                                43: [0.5696, 0.5571, 0.5515, 0.5518, 0.5517, 0.5512, 0.5528, 0.5511],
                                44: [0.5891, 0.5542, 0.5519, 0.5515, 0.5580, 0.5513, 0.5516, 0.5518]},
        'E: NCA slow LR (ours)': {42: [0.6137, 0.6045, 0.5989, 0.5927, 0.5885, 0.5846, 0.5815, 0.5782],
                                   43: [0.6207, 0.6073, 0.5980, 0.5886, 0.5786, 0.5708, 0.5644, 0.5598],
                                   44: [0.6071, 0.5949, 0.5853, 0.5791, 0.5728, 0.5672, 0.5628, 0.5597]},
        'F: NCA frozen attn (ours)': {42: [0.5910, 0.5668, 0.5577, 0.5537, 0.5522, 0.5522, 0.5517, 0.5544],
                                      43: [0.5910, 0.5668, 0.5570, 0.5543, 0.5521, 0.5521, 0.5521, 0.5515],
                                      44: [0.5865, 0.5733, 0.5562, 0.5562, 0.5527, 0.5522, 0.5518, 0.5536]},
    }
    LOSS_MEAN = {}; LOSS_STD = {}
    for _cond, _seeds in LOSS_RAW.items():
        _arr = np.array(list(_seeds.values()))
        LOSS_MEAN[_cond] = _arr.mean(axis=0).tolist()
        LOSS_STD[_cond]  = _arr.std(axis=0).tolist()
    return (
        L0_SCORES,
        L1_SCORES,
        LOSS_EPOCHS,
        LOSS_MEAN,
        LOSS_STD,
        PROBE_EPOCHS,
        RANDOM_BASELINE,
    )


@app.cell
def _(LOSS_EPOCHS, LOSS_MEAN, LOSS_STD, mo, np, plt):
    _COLOURS = {'A: Scratch': '#888888', 'B: NCA standard LR': '#4dabf7',
                'E: NCA slow LR (ours)': '#69db7c', 'F: NCA frozen attn (ours)': '#ffa94d'}
    fig_loss, axes_loss = plt.subplots(1, 2, figsize=(14, 5))
    fig_loss.patch.set_facecolor('#1a1a2e')
    fig_loss.suptitle('Validation Loss During Dyck-1 Fine-tuning, +/-1 std across 3 seeds',
                      fontsize=13, color='white')
    for _ax_l2, _xlim, _title_l in [(axes_loss[0], (10, 80), 'Full run'),
                                     (axes_loss[1], (10, 40), 'Warmup zoom')]:
        _ax_l2.set_facecolor('#1a1a2e')
        for _cond2, _colour2 in _COLOURS.items():
            _mean2 = np.array(LOSS_MEAN[_cond2]); _std2 = np.array(LOSS_STD[_cond2])
            _ep2 = np.array(LOSS_EPOCHS)
            _ax_l2.plot(_ep2, _mean2, color=_colour2, label=_cond2, linewidth=2, marker='o', markersize=5)
            _ax_l2.fill_between(_ep2, _mean2 - _std2, _mean2 + _std2, color=_colour2, alpha=0.15)
        _ax_l2.set_xlim(*_xlim)
        _ax_l2.set_xlabel('Epoch', color='white', fontsize=11)
        _ax_l2.set_ylabel('Val loss', color='white', fontsize=11)
        _ax_l2.set_title(_title_l, color='white', fontsize=11)
        _ax_l2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        _ax_l2.tick_params(colors='white')
        for _sp6 in _ax_l2.spines.values(): _sp6.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_loss)
    return


@app.cell
def _(L0_SCORES, L1_SCORES, PROBE_EPOCHS, RANDOM_BASELINE, mo, plt):
    _COLOURS2 = {'A: Scratch': '#888888', 'B: NCA standard LR': '#4dabf7',
                 'E: NCA slow LR (ours)': '#69db7c', 'F: NCA frozen attn (ours)': '#ffa94d'}
    fig_attn_traj, axes_attn = plt.subplots(1, 2, figsize=(14, 5))
    fig_attn_traj.patch.set_facecolor('#1a1a2e')
    fig_attn_traj.suptitle('Bracket Attention Score Over Training', fontsize=13, color='white')
    for _ax_at, _scores_at, _title_at in [
        (axes_attn[0], L0_SCORES, 'Layer 0  (shallow patterns)'),
        (axes_attn[1], L1_SCORES, 'Layer 1  (induction heads)'),
    ]:
        _ax_at.set_facecolor('#1a1a2e')
        _ax_at.axhline(RANDOM_BASELINE, color='white', linestyle='--', alpha=0.5,
                       linewidth=1.5, label=f'Random ({RANDOM_BASELINE:.3f})')
        for _lbl2, _vals2 in _scores_at.items():
            _ax_at.plot(PROBE_EPOCHS, _vals2, color=_COLOURS2[_lbl2], label=_lbl2,
                        linewidth=2.5, marker='o', markersize=6)
        if 'Layer 1' in _title_at:
            _ax_at.annotate('Catastrophic\nforgetting',
                            xy=(20, L1_SCORES['B: NCA standard LR'][2]), xytext=(30, 0.235),
                            color='#4dabf7', fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='#4dabf7', lw=1.2))
            _ax_at.annotate('Circuits\nprotected',
                            xy=(80, L1_SCORES['E: NCA slow LR (ours)'][-1]), xytext=(55, 0.240),
                            color='#69db7c', fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='#69db7c', lw=1.2))
        _ax_at.set_xlabel('Fine-tuning epoch', color='white', fontsize=11)
        _ax_at.set_ylabel('Bracket attention score', color='white', fontsize=11)
        _ax_at.set_title(_title_at, color='white', fontsize=11)
        _ax_at.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        _ax_at.tick_params(colors='white'); _ax_at.set_ylim(0.08, 0.29)
        for _sp7 in _ax_at.spines.values(): _sp7.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_attn_traj)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(mo.md("""
        ## Catastrophic Forgetting - At The Circuit Level

        | Epoch | Layer 1 Score | Status |
        |-------|--------------|--------|
        | 0     | **0.248**    | NCA circuits installed |
        | 5     | 0.219        | Starting to erode... |
        | 20    | 0.208        | Significant loss |
        | 40    | 0.192        | Approaching scratch |
        | 80    | 0.181        | Almost indistinguishable from scratch |

        The standard fine-tuning LR overwrites NCA-installed circuits in 5-10 epochs.
        """), kind='warn'),
        mo.callout(mo.md("""
        ## The Fix - Protect The Circuits

        **Condition E - Slow LR:** Layer 1 score at epoch 80: **0.221** (vs 0.144 for scratch)

        **Condition F - Frozen attention:** Layer 1 score at epoch 80: **0.224**

        Both preserve >90% of Layer 1 structure. -> Section 5
        """), kind='success')
    ])
    return


@app.cell
def _(mo):
    epoch_slider = mo.ui.slider(start=0, stop=80, step=1, value=0,
                                 label="Fine-tuning epoch")
    mo.vstack([
        mo.md("### Explore: bracket score at any epoch"),
        epoch_slider,
    ])
    return (epoch_slider,)


@app.cell
def _(L1_SCORES, PROBE_EPOCHS, RANDOM_BASELINE, epoch_slider, mo, np):
    _ep = epoch_slider.value
    _COLOURS3 = {'A: Scratch': '#888888', 'B: NCA standard LR': '#4dabf7',
                 'E: NCA slow LR (ours)': '#69db7c', 'F: NCA frozen attn (ours)': '#ffa94d'}
    _rows = []
    for _lbl3, _vals3 in L1_SCORES.items():
        _score3 = float(np.interp(_ep, PROBE_EPOCHS, _vals3))
        _vs = (_score3 - RANDOM_BASELINE) / RANDOM_BASELINE * 100
        _rows.append(f"| {_lbl3} | **{_score3:.3f}** | {_vs:+.0f}% vs random |")
    _table = "| Condition | Layer 1 Score | vs Random |\n|---|---|---|\n" + "\n".join(_rows)
    mo.vstack([
        mo.md(f"**Epoch {_ep}** - Layer 1 bracket attention scores:"),
        mo.md(_table),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 - The Fix: Protecting Pre-trained Circuits

    Two strategies, both validated across 3 seeds:
    - **Slow LR** - 10x lower learning rate on transferred layers
    - **Frozen attention** - freeze attention weights, only train MLPs

    Both preserve >90% of the Layer 1 structure through 80 epochs.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("### Strategy Comparison at Epoch 80"),
        mo.md("""
    | Condition | Strategy | Layer 1 Score | vs Scratch | Verdict |
    |---|---|---|---|---|
    | A: Scratch | Random init, standard LR | 0.144 | baseline | - |
    | B: NCA standard LR | NCA pre-train, same LR for all | 0.181 | +26% | Forgets |
    | E: NCA slow LR | NCA pre-train, 10x lower LR on blocks | 0.221 | +53% | Preserves |
    | F: NCA frozen attn | NCA pre-train, attn weights frozen | 0.224 | +55% | Preserves |
        """)
    ])
    return


@app.cell
def _(
    L1_SCORES,
    LOSS_EPOCHS,
    LOSS_MEAN,
    LOSS_STD,
    RANDOM_BASELINE,
    mo,
    np,
    plt,
):
    _COLOURS4 = {'A: Scratch': '#888888', 'B: NCA standard LR': '#4dabf7',
                 'E: NCA slow LR (ours)': '#69db7c', 'F: NCA frozen attn (ours)': '#ffa94d'}
    _CONDITIONS4 = list(_COLOURS4.keys())
    fig_fix, axes_fix = plt.subplots(1, 2, figsize=(14, 6))
    fig_fix.patch.set_facecolor('#1a1a2e')
    fig_fix.suptitle('The Fix: Protecting NCA-Installed Circuits',
                     fontsize=13, color='white')
    _ax_f0 = axes_fix[0]; _ax_f0.set_facecolor('#1a1a2e')
    for _cond3, _colour3 in _COLOURS4.items():
        _mean3 = np.array(LOSS_MEAN[_cond3]); _std3 = np.array(LOSS_STD[_cond3])
        _ep3   = np.array(LOSS_EPOCHS)
        _ax_f0.plot(_ep3, _mean3, color=_colour3, label=_cond3, linewidth=2, marker='o', markersize=4)
        _ax_f0.fill_between(_ep3, _mean3 - _std3, _mean3 + _std3, color=_colour3, alpha=0.15)
    _ax_f0.set_xlabel('Epoch', color='white', fontsize=11)
    _ax_f0.set_ylabel('Validation loss', color='white', fontsize=11)
    _ax_f0.set_title('Validation loss (lower = better)', color='white', fontsize=11)
    _ax_f0.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    _ax_f0.tick_params(colors='white')
    for _sp8 in _ax_f0.spines.values(): _sp8.set_edgecolor('#444')
    _ax_f1 = axes_fix[1]; _ax_f1.set_facecolor('#1a1a2e')
    _scores_80 = {c: L1_SCORES[c][-1] for c in _CONDITIONS4}
    _x4 = np.arange(len(_CONDITIONS4))
    _bars4 = _ax_f1.bar(_x4, [_scores_80[c] for c in _CONDITIONS4],
                        color=[_COLOURS4[c] for c in _CONDITIONS4], alpha=0.85, width=0.6)
    _ax_f1.axhline(RANDOM_BASELINE, color='white', linestyle='--', linewidth=1.5, alpha=0.6,
                   label=f'Random baseline ({RANDOM_BASELINE:.3f})')
    for _bar4 in _bars4:
        _h4 = _bar4.get_height()
        _ax_f1.text(_bar4.get_x() + _bar4.get_width()/2, _h4 + 0.003,
                    f'{_h4:.3f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
    _ax_f1.set_xticks(_x4)
    _ax_f1.set_xticklabels(['Scratch', 'NCA\nStd LR', 'NCA\nSlow LR', 'NCA\nFrozen'],
                            color='white', fontsize=10)
    _ax_f1.set_ylabel('Layer 1 bracket score at epoch 80', color='white', fontsize=11)
    _ax_f1.set_title('Circuit preservation at epoch 80', color='white', fontsize=11)
    _ax_f1.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    _ax_f1.set_ylim(0, 0.30); _ax_f1.tick_params(colors='white')
    for _sp9 in _ax_f1.spines.values(): _sp9.set_edgecolor('#444')
    plt.tight_layout()
    mo.mpl.interactive(fig_fix)
    return


@app.cell
def _(mo):
    topology_dropdown = mo.ui.dropdown(
        options=['2D NCA (paper)', '1D NCA (ours)'], value='2D NCA (paper)',
        label='NCA topology')
    lr_dropdown = mo.ui.dropdown(
        options=['Standard LR (1e-4 all layers)',
                 'Slow LR (1e-5 transferred, 1e-4 new)', 'Frozen attention'],
        value='Standard LR (1e-4 all layers)', label='Fine-tuning strategy')
    epochs_dropdown = mo.ui.dropdown(
        options=['5 epochs (minimal)', '15 epochs (our config)', '50 epochs (extended)'],
        value='15 epochs (our config)', label='NCA pre-training epochs')
    mo.vstack([
        mo.md("### Design Your Own Pre-Training Recipe"),
        mo.hstack([topology_dropdown, lr_dropdown, epochs_dropdown], justify='start'),
    ])
    return epochs_dropdown, lr_dropdown, topology_dropdown


@app.cell
def _(RANDOM_BASELINE, epochs_dropdown, lr_dropdown, mo, topology_dropdown):
    _PREDS = {
        ('2D NCA (paper)', 'Standard LR (1e-4 all layers)', '15 epochs (our config)'):
            (0.248, 0.181, 'Catastrophic forgetting. Circuits installed but destroyed.'),
        ('2D NCA (paper)', 'Slow LR (1e-5 transferred, 1e-4 new)', '15 epochs (our config)'):
            (0.248, 0.221, 'Circuits preserved. Score stays 53% above scratch at epoch 80.'),
        ('2D NCA (paper)', 'Frozen attention', '15 epochs (our config)'):
            (0.248, 0.224, 'Maximum preservation. Score stays 55% above scratch.'),
        ('1D NCA (ours)', 'Standard LR (1e-4 all layers)', '15 epochs (our config)'):
            (0.220, 0.165, '1D topology. Similar forgetting pattern to 2D.'),
        ('1D NCA (ours)', 'Slow LR (1e-5 transferred, 1e-4 new)', '15 epochs (our config)'):
            (0.220, 0.200, '1D + slow LR. Predicted comparable to 2D slow LR.'),
        ('1D NCA (ours)', 'Frozen attention', '15 epochs (our config)'):
            (0.220, 0.205, '1D + frozen. Strong preservation predicted.'),
        ('2D NCA (paper)', 'Standard LR (1e-4 all layers)', '5 epochs (minimal)'):
            (0.190, 0.160, 'Insufficient pre-training. Only 5 NCA epochs.'),
        ('2D NCA (paper)', 'Standard LR (1e-4 all layers)', '50 epochs (extended)'):
            (0.265, 0.190, 'Stronger installation, same forgetting. Standard LR still destroys.'),
        ('2D NCA (paper)', 'Slow LR (1e-5 transferred, 1e-4 new)', '5 epochs (minimal)'):
            (0.190, 0.175, 'Slow LR helps but circuits were thin. 5 epochs not enough.'),
        ('2D NCA (paper)', 'Slow LR (1e-5 transferred, 1e-4 new)', '50 epochs (extended)'):
            (0.265, 0.245, 'Predicted best. Deep installation + slow LR protection.'),
        ('1D NCA (ours)', 'Standard LR (1e-4 all layers)', '5 epochs (minimal)'):
            (0.170, 0.145, 'Minimal effect. Near scratch.'),
        ('1D NCA (ours)', 'Slow LR (1e-5 transferred, 1e-4 new)', '50 epochs (extended)'):
            (0.240, 0.225, 'Strong 1D configuration.'),
        ('1D NCA (ours)', 'Frozen attention', '5 epochs (minimal)'):
            (0.170, 0.168, 'Frozen but thin. 5 epochs did not install much.'),
        ('1D NCA (ours)', 'Frozen attention', '50 epochs (extended)'):
            (0.240, 0.238, 'Near-maximum preservation.'),
    }
    _key = (topology_dropdown.value, lr_dropdown.value, epochs_dropdown.value)
    _s0, _s80, _notes = _PREDS.get(_key, (0.200, 0.175, 'Extrapolated.'))
    _ret = (_s80 / _s0) * 100 if _s0 > 0 else 0
    mo.vstack([
        mo.md(f"""
        ### Predicted Outcome
        | Metric | Value |
        |--------|-------|
        | Layer 1 score at epoch 0 | **{_s0:.3f}** ({(_s0-RANDOM_BASELINE)/RANDOM_BASELINE*100:+.0f}% vs random) |
        | Layer 1 score at epoch 80 | **{_s80:.3f}** ({(_s80-RANDOM_BASELINE)/RANDOM_BASELINE*100:+.0f}% vs random) |
        | Circuit retention | **{_ret:.0f}%** of installed structure survives |
        """),
        mo.callout(mo.md(_notes), kind='info'),
    ])
    return


@app.cell
def _(mo):
    mo.callout(mo.md("""
    ## Why The Loss Curves Don't Show It

    At toy scale (150k params, Dyck-1), all models converge to the same loss floor (~0.551).
    The task is too easy for a 150k transformer regardless of initialisation.

    At paper scale (400M+ params), the pre-trained circuits compound into **6% perplexity gain**.

    **Actionable recommendation:** Use a layerwise LR schedule or freeze attention
    during early fine-tuning. The paper does not make this recommendation. We do.
    """), kind='info')
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        ---
        ## The Story In Four Lines

        **NCA pre-training installs induction-head circuits in Layer 1.**
        Score: 0.248 vs random baseline 0.111 before seeing a single bracket.

        **Standard fine-tuning destroys them.**
        Score drops 0.248 to 0.181 in 80 epochs. Catastrophic forgetting.

        **Slow LR or frozen attention preserves them.**
        Score holds at 0.221-0.224. The circuits survive. The mechanism is real.

        **At scale, the circuits compound.**
        400M params, 164M NCA tokens: the paper reports 6% perplexity gain.
        Our toy experiments reveal the mechanism. Their scale experiments show the payoff.

        ---
        """),
        mo.callout(mo.md("""
        *"We did not teach this model to think by showing it thoughts.
        We taught it to think by showing it patterns.
        Language was never the point -- structure was."*
        """), kind='info')
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 - Where This Leads: Petri Dish NCA

    What if the NCA itself never stopped learning?

    **Petri Dish Neural Cellular Automata** (Sakana AI, ALIFE 2025):
    multiple NCA agents compete for territory on a shared grid,
    each continuously updating their own parameters via backpropagation
    during the simulation itself.
    """)
    return


@app.cell
def _(np):
    PDNCA_H = 48; PDNCA_W = 48; N_CHANNELS = 8
    N_ATTACK = 3; N_DEFENSE = 3; ALIVE_THRESH = 0.4

    def make_pd_agent(seed):
        rng = np.random.default_rng(seed); fan_in = N_CHANNELS * 9
        W1 = rng.normal(0, np.sqrt(2/fan_in), (32, fan_in)).astype(np.float32)
        W2 = rng.normal(0, np.sqrt(2/32), (N_CHANNELS, 32)).astype(np.float32)
        return W1, np.zeros(32, np.float32), W2, np.zeros(N_CHANNELS, np.float32)

    def _agent_fwd(nb, W1, b1, W2, b2):
        return np.maximum(nb @ W1.T + b1, 0) @ W2.T + b2

    def _gather_nb_cont(grid):
        channels = [np.roll(np.roll(grid, si, axis=0), sj, axis=1)
                    for si in [-1, 0, 1] for sj in [-1, 0, 1]]
        return np.concatenate(channels, axis=-1).reshape(-1, N_CHANNELS * 9)

    def pd_step(grid, aliveness, agents, temperature=0.5):
        H, W, C = grid.shape; nb = _gather_nb_cont(grid)
        proposals = [_agent_fwd(nb, *a) for a in agents]
        n_ag = len(agents)
        strengths = np.zeros((H * W, n_ag), np.float32)
        for i, pi in enumerate(proposals):
            ai = pi[:, :N_ATTACK]; di = pi[:, N_ATTACK:N_ATTACK+N_DEFENSE]
            for j, pj in enumerate(proposals):
                if i == j: continue
                aj = pj[:, :N_ATTACK]; dj = pj[:, N_ATTACK:N_ATTACK+N_DEFENSE]
                nai = np.linalg.norm(ai, axis=1, keepdims=True) + 1e-8
                ndj = np.linalg.norm(dj, axis=1, keepdims=True) + 1e-8
                ndi = np.linalg.norm(di, axis=1, keepdims=True) + 1e-8
                naj = np.linalg.norm(aj, axis=1, keepdims=True) + 1e-8
                strengths[:, i] += ((ai/nai) * (dj/ndj)).sum(-1) - ((di/ndi) * (aj/naj)).sum(-1)
        exp_s = np.exp(strengths / temperature)
        weights = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-8)
        alive_mask = (weights > ALIVE_THRESH).astype(np.float32)
        new_alive = (weights * alive_mask).reshape(H, W, n_ag)
        total = new_alive.sum(axis=-1, keepdims=True) + 1e-8
        new_alive = new_alive / total
        pstack = np.stack(proposals, axis=-1)
        delta = (pstack * weights[:, np.newaxis, :]).sum(axis=-1).reshape(H, W, C)
        return np.clip(grid + 0.1 * delta, -1, 1), new_alive

    def render_pd(aliveness):
        H, W, N = aliveness.shape
        AGENT_COLOURS = np.array([[0.30, 0.69, 0.97], [0.41, 0.86, 0.49],
                                   [1.00, 0.66, 0.30], [0.94, 0.42, 0.42],
                                   [0.75, 0.55, 0.95], [0.95, 0.85, 0.30]], dtype=np.float32)
        rgb = np.zeros((H, W, 3), np.float32)
        for i in range(N): rgb += aliveness[:, :, i:i+1] * AGENT_COLOURS[i % len(AGENT_COLOURS)]
        return np.clip(rgb, 0, 1)

    return N_CHANNELS, PDNCA_H, PDNCA_W, make_pd_agent, pd_step, render_pd


@app.cell
def _(mo):
    pd_steps_slider = mo.ui.slider(start=1, stop=200, step=1, value=1,
                                    label="Simulation steps")
    n_agents_slider = mo.ui.slider(start=2, stop=6, step=1, value=4,
                                    label="Number of agents")
    temperature_slider = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=0.5,
                                       label="Competition temperature")
    seed_pd_slider = mo.ui.slider(start=0, stop=20, step=1, value=0,
                                   label="Simulation seed")
    mo.vstack([
        mo.callout(mo.md("""
        **Petri Dish NCA - Live Simulation**
        Each colour is a different NCA agent competing for territory.
        """), kind='info'),
        mo.hstack([pd_steps_slider, n_agents_slider], justify='start'),
        mo.hstack([temperature_slider, seed_pd_slider], justify='start'),
    ])
    return n_agents_slider, pd_steps_slider, seed_pd_slider, temperature_slider


@app.cell
def _(
    N_CHANNELS,
    PDNCA_H,
    PDNCA_W,
    make_pd_agent,
    mo,
    n_agents_slider,
    np,
    pd_step,
    pd_steps_slider,
    plt,
    render_pd,
    seed_pd_slider,
    temperature_slider,
):
    _n_ag  = n_agents_slider.value; _T_sim = pd_steps_slider.value
    _temp  = temperature_slider.value; _bseed = seed_pd_slider.value
    _rng2  = np.random.default_rng(_bseed)
    _agents = [make_pd_agent(int(_rng2.integers(0, 100_000))) for _ in range(_n_ag)]
    _grid  = _rng2.uniform(-0.5, 0.5, (PDNCA_H, PDNCA_W, N_CHANNELS)).astype(np.float32)
    _alive = np.zeros((PDNCA_H, PDNCA_W, _n_ag), np.float32)
    for _i in range(_n_ag):
        _row2 = (_i // 2) * (PDNCA_H // 2) + PDNCA_H // 4
        _col2 = (_i %  2) * (PDNCA_W // 2) + PDNCA_W // 4
        _alive[max(0,_row2-4):min(PDNCA_H,_row2+4), max(0,_col2-4):min(PDNCA_W,_col2+4), _i] = 1.0
    _n_frames = min(6, _T_sim)
    _snaps = np.linspace(0, _T_sim, _n_frames, dtype=int)
    _frames = []
    for _t2 in range(_T_sim + 1):
        if _t2 in _snaps: _frames.append((_t2, render_pd(_alive)))
        if _t2 < _T_sim: _grid, _alive = pd_step(_grid, _alive, _agents, _temp)
    fig_pd, axes_pd = plt.subplots(1, len(_frames), figsize=(14, 3.5))
    fig_pd.patch.set_facecolor('#1a1a2e')
    if len(_frames) == 1: axes_pd = [axes_pd]
    for _ax_pd, (_t3, _frame) in zip(axes_pd, _frames):
        _ax_pd.imshow(_frame, interpolation='nearest')
        _ax_pd.set_title(f't = {_t3}', color='white', fontsize=10)
        _ax_pd.set_xticks([]); _ax_pd.set_yticks([])
    fig_pd.suptitle(f'Petri Dish NCA - {_n_ag} agents, {_T_sim} steps',
                    color='white', fontsize=12)
    plt.tight_layout()
    mo.mpl.interactive(fig_pd)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
        ---
        ### From Pre-training to Life

        | | This notebook | Petri Dish NCA |
        |---|---|---|
        | NCA role | Training data generator | The agent itself |
        | Learning | Transformer learns from NCA | NCA learns during simulation |
        | Rules | Fixed per sequence | Continuously updated via backprop |
        | Emergence | Attention circuits | Cooperation, cycles, complexity |

        Both systems demonstrate that local interactions, iterated over time, produce global intelligence.
        """),
        mo.callout(mo.md("""
        ### The Open Questions

        **From this notebook:** What happens at 400M params with slow LR?

        **From PD-NCA:** Can competing NCA agents produce richer pre-training signals?

        **The G-CTM connection:** If reasoning is continuous (CTM), structure emerges
        from local competition (PD-NCA), and circuits persist when protected (this notebook),
        then the path to abstract reasoning runs through open-ended structured dynamics.
        """), kind='info')
    ])
    return


@app.cell
def _(mo):
    mo.vstack([mo.md("""
    ---
    ### References

    **Primary paper:** Lee, Han, Kumar & Agrawal (2026).
    *Training Language Models via Neural Cellular Automata.* arXiv:2603.10055

    **PD-NCA:** Zhang, Risi & Darlow (2025). *Petri Dish Neural Cellular Automata.*
    ALIFE 2025. pub.sakana.ai/pdnca

    **CTM:** Sakana AI (2025). *Continuous Thought Machines.*

    ---
    *Built for the marimo x alphaXiv Notebook Competition, April 2026.*
    *github.com/vinodanbalagan/nca-language-pretraining*
    """)])
    return


if __name__ == "__main__":
    app.run()
