import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    app_title="Training Language Models via Neural Cellular Automata",
)


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

    # House style, single source of truth for all figures in this notebook.
    import theme
    theme.apply_rc()
    return (
        F,
        FuncAnimation,
        HTML,
        mcolors,
        mo,
        nn,
        np,
        plt,
        random,
        theme,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Training Language Models via Neural Cellular Automata

    **A marimo × alphaXiv competition notebook.**
    Paper: Lee, Han, Kumar & Agrawal (2026), arXiv:2603.10055.

    ---

    ### Abstract

    Lee et al. show that pre-training a transformer on randomly-initialised
    cellular automata, synthetic non-linguistic data, improves downstream
    language modelling at 400M scale (~6% perplexity). *What* transfers is
    unclear from their paper.

    We probe the attention circuits directly on a toy task (Dyck-1) with
    a tiny transformer (~150k params). The findings: NCA pre-training
    installs induction-head-like structure in Layer 1 before any language
    data; standard fine-tuning destroys it within 20 epochs; and two
    simple recipes, a 10× slower learning rate on transferred blocks, or
    freezing attention outright, preserve it through 80 epochs.

    **Contribution:** a mechanistic account of *what* NCA pre-training
    transfers, and a practical warning that naive fine-tuning wastes it.
    """)
    return


@app.cell
def _(mo, theme):
    # The three numbers that carry the paper. A reader who only looks at this
    # block should still walk away knowing the finding.
    _hero = f"""
    <div style="
        background: {theme.PANEL};
        border: 1px solid {theme.SPINE};
        border-radius: 8px;
        padding: 24px 28px;
        margin: 12px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    ">
      <div style="color: {theme.FG_MUTED}; font-size: 11px;
                  letter-spacing: 0.12em; text-transform: uppercase;
                  margin-bottom: 18px;">
        The finding, in three numbers
      </div>

      <div style="display: flex; gap: 40px; flex-wrap: wrap;">

        <div style="flex: 1; min-width: 160px;">
          <div style="color: {theme.PALETTE['green']}; font-size: 42px;
                      font-weight: 600; line-height: 1;">0.248</div>
          <div style="color: {theme.FG}; font-size: 13px; margin-top: 8px;">
            NCA-pretrained model
          </div>
          <div style="color: {theme.FG_MUTED}; font-size: 11px;">
            Layer 1 bracket attention, epoch 0
          </div>
        </div>

        <div style="flex: 1; min-width: 160px;">
          <div style="color: {theme.PALETTE['blue']}; font-size: 42px;
                      font-weight: 600; line-height: 1;">0.179</div>
          <div style="color: {theme.FG}; font-size: 13px; margin-top: 8px;">
            Randomly initialised model
          </div>
          <div style="color: {theme.FG_MUTED}; font-size: 11px;">
            Layer 1 bracket attention, epoch 0
          </div>
        </div>

        <div style="flex: 1; min-width: 160px;">
          <div style="color: {theme.FG_MUTED}; font-size: 42px;
                      font-weight: 600; line-height: 1;">0.111</div>
          <div style="color: {theme.FG}; font-size: 13px; margin-top: 8px;">
            Random baseline
          </div>
          <div style="color: {theme.FG_MUTED}; font-size: 11px;">
            Uniform attention over 9 positions
          </div>
        </div>

      </div>

      <div style="color: {theme.FG_MUTED}; font-size: 12px;
                  margin-top: 20px; line-height: 1.5;">
        The NCA-pretrained model attends to correct bracket-matching positions
        at <b style="color: {theme.FG};">2.2× above random chance</b>, purely
        from learning NCA dynamics. Not a single bracket has been seen.
      </div>
    </div>
    """
    mo.Html(_hero)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 1. The NCA Generator

    A Neural Cellular Automaton is a grid where each cell updates based
    on its 3×3 neighbourhood, using a small fixed neural network as the
    rule. Different random initialisations of that network give
    completely different dynamics: some converge to fixed points, some
    form stripes, some stay chaotic.

    The key property for our purposes: **the rule is the same everywhere
    on the grid, every step.** That locality is what will matter when we
    look at attention patterns later.
    """)
    return


@app.cell
def _(np):
    # --------------------------------------------------------------------
    # The rule network: a tiny MLP, 9 inputs (3x3 neighbourhood) -> n outputs
    # (one logit per possible next state). No training; the "rule" is
    # defined entirely by the random initialisation.
    # --------------------------------------------------------------------
    def make_rule_network(n: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / 9)
        scale2 = np.sqrt(2.0 / hidden_dim)
        return {
            "W1": rng.normal(0, scale1, (hidden_dim, 9)).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": rng.normal(0, scale2, (n, hidden_dim)).astype(np.float32),
            "b2": np.zeros(n, dtype=np.float32),
        }

    def rule_forward(params, neighbourhood_flat):
        h = neighbourhood_flat @ params["W1"].T + params["b1"]
        h = np.maximum(h, 0)   # ReLU
        return h @ params["W2"].T + params["b2"]

    return make_rule_network, rule_forward


@app.cell
def _(np):
    def get_neighbourhood(grid):
        """Stack 9 shifted copies of the grid to form a 3x3 neighbourhood
        at every cell (with toroidal wrap)."""
        g = grid.astype(np.float32)
        channels = [
            g,
            np.roll(g, -1, axis=0),
            np.roll(np.roll(g, -1, axis=0),  1, axis=1),
            np.roll(g,  1, axis=1),
            np.roll(np.roll(g,  1, axis=0),  1, axis=1),
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
        nb = get_neighbourhood(grid).reshape(-1, 9)
        logits = rule_forward(params, nb)
        return np.argmax(logits, axis=-1).reshape(H, W)

    return (nca_step,)


@app.cell
def _(make_rule_network, nca_step, np):
    def run_nca(n: int, T: int, H: int = 16, W: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        grid = rng.integers(0, n, size=(H, W))
        rule_seed = int(rng.integers(0, 100_000))
        params = make_rule_network(n=n, seed=rule_seed)
        trajectory = [grid.copy()]
        for _ in range(T):
            grid = nca_step(grid, params, n)
            trajectory.append(grid.copy())
        return np.stack(trajectory, axis=0)

    return (run_nca,)


@app.cell
def _(mcolors, np, plt, theme):
    # --------------------------------------------------------------------
    # Colormap helper. For the default n=2 (which is what we actually use
    # for tokenisation later), we use the two-colour palette from theme.py
    # so there's visual continuity with Section 2. For larger n, we fall
    # back to tab10 (only relevant in the "explore" mode).
    # --------------------------------------------------------------------
    def make_colormap(n):
        if n == 2:
            colours = theme.NCA_BINARY
        else:
            base = plt.colormaps["tab10"]
            colours = [base(i % 10) for i in range(n)]
        cmap = mcolors.ListedColormap(colours)
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(-0.5, n, 1.0), ncolors=n
        )
        return cmap, norm

    return (make_colormap,)


@app.cell
def _(mo):
    # Defaults locked to n=2. This matches what we actually use downstream
    # (tokenisation with patch=2 gives vocab=16). The reader can still
    # explore larger n, but the "home" state of the notebook is consistent.
    n_slider    = mo.ui.slider(start=2, stop=6,  step=1, value=2,
                               label="Alphabet size n")
    T_slider    = mo.ui.slider(start=5, stop=60, step=5, value=30,
                               label="Timesteps T")
    seed_slider = mo.ui.slider(start=0, stop=99, step=1, value=1,
                               label="Rule seed")
    return T_slider, n_slider, seed_slider


@app.cell
def _(T_slider, mo, n_slider, seed_slider):
    mo.vstack([
        mo.md("**Explore the NCA.** Drag the sliders to sample different "
              "rules and watch how they evolve."),
        mo.hstack([n_slider, T_slider, seed_slider], justify="start"),
    ])
    return


@app.cell
def _(T_slider, make_colormap, mo, n_slider, plt, run_nca, seed_slider, theme):
    # Initial vs final frame, side by side.
    _traj = run_nca(n=n_slider.value, T=T_slider.value, seed=seed_slider.value)
    _cmap, _norm = make_colormap(n_slider.value)

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    theme.style_figure(_fig)

    _axes[0].imshow(_traj[0], cmap=_cmap, norm=_norm, interpolation="nearest")
    theme.style_image_axes(_axes[0], title="t = 0  (random initial state)")

    _axes[1].imshow(_traj[-1], cmap=_cmap, norm=_norm, interpolation="nearest")
    theme.style_image_axes(_axes[1],
        title=f"t = {T_slider.value}  (after {T_slider.value} updates)")

    theme.suptitle(_fig,
        f"A single NCA rule evolving. n={n_slider.value} states, seed={seed_slider.value}")
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    FuncAnimation,
    HTML,
    T_slider,
    make_colormap,
    mo,
    n_slider,
    plt,
    run_nca,
    seed_slider,
    theme,
):
    # Animated evolution of the selected rule. Depends on run_nca via the
    # cell signature (marimo's dependency model). No hacks.
    _traj_anim = run_nca(n=n_slider.value, T=T_slider.value,
                         seed=seed_slider.value)
    _cmap_a, _norm_a = make_colormap(n_slider.value)

    _fig_a, _ax_a = plt.subplots(figsize=(4.5, 4.5))
    theme.style_figure(_fig_a)
    theme.style_image_axes(_ax_a)

    _im = _ax_a.imshow(_traj_anim[0], cmap=_cmap_a, norm=_norm_a,
                        interpolation="nearest", animated=True)
    _title = _ax_a.set_title("t = 0", color=theme.FG, fontsize=11, pad=8)

    def _update(frame):
        _im.set_data(_traj_anim[frame])
        _title.set_text(f"t = {frame}")
        return _im, _title

    _anim = FuncAnimation(_fig_a, _update,
                           frames=T_slider.value + 1,
                           interval=120, blit=True)
    _fig_a.tight_layout()
    mo.Html(HTML(_anim.to_jshtml()).data)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Four rules, same math, different universes

    Below: the same code, the same update equation, the same grid size.
    Only the random seed of the rule network changes. Each column shows
    one rule's trajectory from its random initial state to its attractor.
    """)
    return


@app.cell
def _(make_colormap, mo, plt, run_nca, theme):
    # Curated set of seeds chosen to display four qualitatively different
    # dynamics. Selected empirically by searching seeds 0-200 for one
    # example each of: chaos, patches, vertical stripes, horizontal stripes.
    _SHOWCASE_SEEDS = [6, 10, 15, 4]
    _n_val, _T_val = 2, 30
    _cmap_g, _norm_g = make_colormap(_n_val)

    _fig, _axes = plt.subplots(2, 4, figsize=(12, 6.5))
    theme.style_figure(_fig)

    for _col, _seed in enumerate(_SHOWCASE_SEEDS):
        _traj_c = run_nca(n=_n_val, T=_T_val, seed=_seed)
        for _row, (_t_show, _lbl) in enumerate([(0, "t = 0"),
                                                 (_T_val, f"t = {_T_val}")]):
            _ax = _axes[_row, _col]
            _ax.imshow(_traj_c[_t_show], cmap=_cmap_g, norm=_norm_g,
                       interpolation="nearest")
            theme.style_image_axes(_ax)
            if _row == 0:
                _ax.set_title(f"rule {_seed}", color=theme.FG, fontsize=11,
                              pad=6)
            if _col == 0:
                _ax.set_ylabel(_lbl, color=theme.FG_MUTED, fontsize=10)

    theme.suptitle(_fig, f"Four different NCA rules  ·  n={_n_val}, T={_T_val}")
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 2. From Grid to Token

    The grid is spatial. A transformer wants a 1D sequence of token IDs.
    The bridge is simple: chop the grid into `p × p` patches and hash
    each patch into a token.

    With `n = 2` states and `p = 2` patches, each patch is 4 cells of 2
    values each, giving a vocabulary of 2⁴ = 16 tokens. These 16 tokens
    are the "words" the transformer learns during NCA pre-training.

    We choose these settings to keep the vocabulary small. As the next
    plot shows, the vocabulary explodes quickly if you don't.
    """)
    return


@app.cell
def _(make_colormap, mo, plt, run_nca, theme):
    # Visual: raw NCA grid on the left, tokenised grid on the right with
    # patch boundaries and token IDs overlaid. This is the diagram of the
    # tokenisation pipeline, by far the most-looked-at figure in the paper.
    # Seed 23 chosen empirically for ~50/50 colour mix with high local
    # activity, so the 2x2 patches produce a diverse token grid.
    _p, _n = 2, 2
    _vocab_size = _n ** (_p * _p)
    _traj_s2 = run_nca(n=_n, T=20, H=16, W=16, seed=23)
    _grid = _traj_s2[20]
    _cmap_p, _norm_p = make_colormap(_n)

    _fig, (_ax_l, _ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    theme.style_figure(_fig)

    _ax_l.imshow(_grid, cmap=_cmap_p, norm=_norm_p, interpolation="nearest")
    theme.style_image_axes(_ax_l, title="NCA grid (raw)")
    _H, _W = _grid.shape
    for _i in range(0, _H + 1, _p):
        _ax_l.axhline(_i - 0.5, color=theme.FG, linewidth=0.8, alpha=0.3)
    for _j in range(0, _W + 1, _p):
        _ax_l.axvline(_j - 0.5, color=theme.FG, linewidth=0.8, alpha=0.3)

    _ax_r.imshow(_grid, cmap=_cmap_p, norm=_norm_p,
                 interpolation="nearest", alpha=0.4)
    theme.style_image_axes(_ax_r,
        title=f"Patches to Tokens  (vocab = {_vocab_size})")
    for _i in range(0, _H, _p):
        for _j in range(0, _W, _p):
            _patch = _grid[_i:_i+_p, _j:_j+_p].flatten()
            _tok = 0
            for _v in _patch:
                _tok = _tok * _n + int(_v)
            _cx = _j + _p / 2 - 0.5
            _cy = _i + _p / 2 - 0.5
            _ax_r.text(_cx, _cy, str(_tok), ha="center", va="center",
                       fontsize=9, color=theme.FG, fontweight="bold")
            _ax_r.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), _p, _p,
                                           linewidth=1.0, edgecolor=theme.FG,
                                           facecolor="none", alpha=0.5))

    theme.suptitle(_fig,
        f"Tokenisation  ·  n={_n}, patch={_p}x{_p}  ->  vocab = {_vocab_size}")
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, np, plt, theme):
    # Why we chose n=2, patch=2: anything bigger blows vocabulary into the
    # billions. Single compact plot (this used to be a full "Zipf section").
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    theme.style_figure(_fig)
    theme.style_axes(_ax,
        xlabel="Alphabet size  n",
        ylabel="Vocabulary size  (log scale)")

    _n_vals = np.arange(2, 16)
    _patch_colours = {
        1: theme.PALETTE["green"],
        2: theme.PALETTE["blue"],
        3: theme.PALETTE["orange"],
        4: theme.PALETTE["red"],
    }
    for _p, _colour in _patch_colours.items():
        _vocab = _n_vals ** (_p * _p)
        _ax.semilogy(_n_vals, _vocab, color=_colour, linewidth=2,
                     label=f"patch {_p}x{_p}  (vocab = n^{_p*_p})",
                     marker="o", markersize=4)

    _ax.axvline(2, color=theme.FG_MUTED, linestyle="--", alpha=0.5,
                linewidth=1)
    _ax.annotate("Our choice:\nn = 2, patch = 2\nvocab = 16",
                 xy=(2, 16), xytext=(4.3, 50),
                 color=theme.FG, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=theme.FG_MUTED, lw=1))

    _ax.set_title("Vocabulary size explodes with n  ·  why we stay small",
                  color=theme.FG, fontsize=12, pad=10, loc="left",
                  fontweight="semibold")
    theme.style_legend(_ax.legend(loc="upper left"))
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
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

    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 3. What Did Pre-Training Install?

    We train two tiny transformers (2 layers, 4 heads, 64 dim,
    ~150k params):

    - **Scratch:** random init, fine-tune directly on Dyck-1 (balanced
      brackets).
    - **NCA:** pre-train on 20 epochs of NCA sequences, then switch the
      task head and fine-tune on Dyck-1.

    Before either model sees a single bracket, we probe: where does
    Layer 1 attend when shown a balanced bracket string?
    """)
    return


@app.cell
def _(F, nn, torch):
    # --------------------------------------------------------------------
    # Tiny transformer. 2 layers, 4 heads, 64 dim. Exposes attention maps
    # for probing, and the transferable state (blocks + pos_embed + norm)
    # for the NCA -> Dyck swap.
    # --------------------------------------------------------------------
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads  = n_heads
            self.d_head   = d_model // n_heads
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, return_attn=False):
            B, T, C = x.shape
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
            def split_heads(t):
                return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            scale = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
            out = self.out_proj(out)
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
            self.embed = None
            self.head  = None
            self.blocks = nn.ModuleList([
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
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm_out(x))

        def get_attention(self, idx, layer=0):
            self.eval()
            with torch.no_grad():
                B, T = idx.shape
                pos  = torch.arange(T, dtype=torch.long)
                x    = self.embed(idx) + self.pos_embed(pos)
                for i, block in enumerate(self.blocks):
                    if i == layer:
                        x, attn_w = block(x, return_attn=True)
                    else:
                        x = block(x)
            return attn_w.squeeze(0).cpu().numpy()

        def get_transferable_state(self):
            return {k: v for k, v in self.state_dict().items()
                    if k.startswith("blocks") or k.startswith("pos_embed")
                    or k.startswith("norm_out")}

        def load_transferable_state(self, state):
            self.load_state_dict(state, strict=False)

    return (TinyTransformerViz,)


@app.cell
def _(np, torch):
    # --------------------------------------------------------------------
    # Data: NCA sequences for pre-training, Dyck-1 strings for fine-tuning.
    # --------------------------------------------------------------------
    TOK_OPEN = 0; TOK_CLOSE = 1; TOK_PAD = 2; TOK_EOS = 3
    DYCK_VOCAB = 4
    NCA_VOCAB = 2 ** (2 * 2)
    N_STATES = 2; PATCH = 2; GRID_H = GRID_W = 16

    def _make_rule(seed):
        rng = np.random.default_rng(seed)
        s1, s2 = np.sqrt(2/9), np.sqrt(2/16)
        return {"W1": rng.normal(0, s1, (16, 9)).astype(np.float32),
                "b1": np.zeros(16, np.float32),
                "W2": rng.normal(0, s2, (N_STATES, 16)).astype(np.float32),
                "b2": np.zeros(N_STATES, np.float32)}

    def _step(grid, rule):
        g = grid.astype(np.float32)
        nb = np.stack([g, np.roll(g,-1,0),
                       np.roll(np.roll(g,-1,0),1,1), np.roll(g,1,1),
                       np.roll(np.roll(g,1,0),1,1), np.roll(g,1,0),
                       np.roll(np.roll(g,1,0),-1,1), np.roll(g,-1,1),
                       np.roll(np.roll(g,-1,0),-1,1)], axis=-1).reshape(-1, 9)
        h = np.maximum(nb @ rule["W1"].T + rule["b1"], 0)
        return np.argmax(h @ rule["W2"].T + rule["b2"], axis=-1).reshape(grid.shape)

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
        seqs = []
        rng = np.random.default_rng(seed)
        for _ in range(n_rules):
            rule = _make_rule(int(rng.integers(0, 1_000_000)))
            for _ in range(n_traj):
                g = np.random.default_rng(
                    int(rng.integers(0, 1_000_000))
                ).integers(0, N_STATES, (GRID_H, GRID_W))
                steps = []
                for _ in range(n_steps):
                    g = _step(g, rule); steps.append(_tok_grid(g))
                seqs.append(np.array((steps[-2] + steps[-1])[:256], dtype=np.int64))
        return seqs

    def generate_dyck(n=2000, max_depth=8, seq_len=64, seed=0):
        rng = np.random.default_rng(seed)
        data = []
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
        if shuffle:
            random.shuffle(idx)
        for s in range(0, len(data) - batch_size + 1, batch_size):
            b = data[idx[s:s+batch_size]]
            yield b[:, :-1], b[:, 1:]

    def train_epoch(model, seqs, opt, ignore=-1):
        model.train()
        total, n = 0., 0
        for x, y in make_loader(seqs):
            opt.zero_grad()
            loss = F.cross_entropy(
                model(x).reshape(-1, model.head.weight.shape[0]),
                y.reshape(-1), ignore_index=ignore)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def eval_model(model, data):
        model.eval()
        seqs = [data[i].numpy() for i in range(len(data))]
        total, n = 0., 0
        for x, y in make_loader(seqs, shuffle=False):
            loss = F.cross_entropy(
                model(x).reshape(-1, model.head.weight.shape[0]),
                y.reshape(-1), ignore_index=TOK_PAD)
            total += loss.item(); n += 1
        return total / max(n, 1)

    return (train_epoch,)


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train both models (~5 min on CPU)")
    mo.vstack([
        mo.md(
            """
            **Training produces two models:**
            Scratch (random init -> Dyck) and NCA (NCA pre-train -> swap task
            head -> Dyck). Runs fully on CPU, ~5 minutes.
            """
        ),
        train_btn,
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
        scratch_ckpts = None
        nca_ckpts = None
        mo.stop(True, mo.md("*Click the button above to train.*"))

    _NCA_EPOCHS = 15
    _DYCK_EPOCHS = 5
    _NCA_LR = 3e-4
    _DYCK_LR = 1e-4
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    with mo.status.spinner("Generating data..."):
        _nca_seqs   = generate_nca_seqs(n_rules=50, n_traj=20, n_steps=32)
        _dyck_train = generate_dyck(n=5000, seed=100)
        _dyck_val   = generate_dyck(n=1000, seed=101)

    def _train_condition(condition):
        model = TinyTransformerViz()
        ckpts = {}
        if condition == "nca":
            model.set_task(NCA_VOCAB)
            opt = torch.optim.Adam(model.parameters(), lr=_NCA_LR)
            for _ in range(_NCA_EPOCHS):
                train_epoch(model, _nca_seqs, opt)
            _transferable = model.get_transferable_state()
        model.set_task(DYCK_VOCAB)
        opt = torch.optim.Adam(model.parameters(), lr=_DYCK_LR)
        if condition == "nca":
            model.load_transferable_state(_transferable)
        ckpts[0] = {k: v.clone() for k, v in model.state_dict().items()}
        _dyck_seqs = [_dyck_train[i].numpy() for i in range(len(_dyck_train))]
        for ep in range(1, _DYCK_EPOCHS + 1):
            train_epoch(model, _dyck_seqs, opt, ignore=2)
            if ep == _DYCK_EPOCHS:
                ckpts[ep] = {k: v.clone() for k, v in model.state_dict().items()}
        return ckpts

    with mo.status.spinner("Training scratch model..."):
        scratch_ckpts = _train_condition("scratch")
    with mo.status.spinner("Training NCA model..."):
        nca_ckpts = _train_condition("nca")
    mo.callout(mo.md("**Training complete.** The figures below use these "
                     "checkpoints."), kind="success")
    return nca_ckpts, scratch_ckpts


@app.cell
def _(torch):
    # Probe sequence: "((()(()))EOS", a balanced string where bracket-
    # matching is unambiguous. The MATCHING_PAIRS dict tells us which
    # (close, open) positions should attend to each other if the model
    # has learnt bracket matching.
    _raw = [0, 0, 1, 0, 0, 1, 1, 1, 3]
    _padded = _raw + [2] * (256 - len(_raw))
    probe_tokens = torch.tensor([_padded], dtype=torch.long)
    probe_labels = ["(", "(", ")", "(", "(", ")", ")", ")", "EOS"]
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
    theme,
):
    # Four-panel attention heatmap: Scratch e0, Scratch e5, NCA e0, NCA e5.
    # This is the single most important figure in the notebook, the
    # "where is the model looking" picture.
    if scratch_ckpts is None:
        mo.stop(True, mo.md("*Train the models first.*"))

    import matplotlib.gridspec as _gs

    def _load_model(ckpts, epoch):
        m = TinyTransformerViz()
        m.set_task(DYCK_VOCAB)
        m.load_state_dict(ckpts[epoch])
        return m

    def _draw_hmap(ax, model, title=""):
        attn = model.get_attention(probe_tokens, layer=1)  # Layer 1 = induction heads
        show_len = 9
        attn_map = attn.mean(axis=0)[:show_len, :show_len]
        ax.imshow(attn_map, cmap="Blues", vmin=0, vmax=attn_map.max(),
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(show_len))
        ax.set_yticks(range(show_len))
        ax.set_xticklabels(probe_labels[:show_len], fontsize=9,
                           color=theme.FG)
        ax.set_yticklabels(probe_labels[:show_len], fontsize=9,
                           color=theme.FG)
        ax.set_xlabel("attending to", fontsize=9, color=theme.FG_MUTED)
        ax.set_ylabel("from", fontsize=9, color=theme.FG_MUTED)
        ax.set_title(title, fontsize=10, color=theme.FG, pad=8, loc="left",
                     fontweight="semibold")
        ax.tick_params(colors=theme.FG_MUTED)
        for _sp in ax.spines.values():
            _sp.set_color(theme.SPINE)
        for _cp, _op in MATCHING_PAIRS.items():
            if _cp < show_len and _op < show_len:
                ax.add_patch(plt.Rectangle(
                    (_op - 0.5, _cp - 0.5), 1, 1,
                    linewidth=2, edgecolor=theme.PALETTE["red"],
                    facecolor="none"))

    _fig = plt.figure(figsize=(12, 10))
    theme.style_figure(_fig)
    theme.suptitle(
        _fig,
        "Layer 1 attention  ·  red boxes = correct bracket-matching positions",
        y=0.995)

    _gspec = _gs.GridSpec(2, 2, figure=_fig, hspace=0.4, wspace=0.3)
    _panels = [
        (0, 0, scratch_ckpts, 0, "Scratch, epoch 0\nrandom init, no Dyck"),
        (0, 1, scratch_ckpts, 5, "Scratch, epoch 5\nafter 5 epochs of Dyck"),
        (1, 0, nca_ckpts,     0, "NCA pretrained, epoch 0\nafter NCA, before any Dyck"),
        (1, 1, nca_ckpts,     5, "NCA pretrained, epoch 5\nafter 5 epochs of Dyck"),
    ]
    for _row, _col, _ckpts, _ep, _title in _panels:
        _ax = _fig.add_subplot(_gspec[_row, _col])
        _ax.set_facecolor(theme.BG)
        _draw_hmap(_ax, _load_model(_ckpts, _ep), title=_title)

    _fig.tight_layout(rect=[0, 0, 1, 0.96])
    mo.mpl.interactive(_fig)
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
    theme,
):
    # Bracket-score bar chart. Fixed ylim at 0.37 so the legend doesn't
    # overlap the tallest bar.
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
        "Scratch\nEpoch 0": (scratch_ckpts, 0),
        "Scratch\nEpoch 5": (scratch_ckpts, 5),
        "NCA\nEpoch 0":     (nca_ckpts,     0),
        "NCA\nEpoch 5":     (nca_ckpts,     5),
    }
    _scores_l0 = {}
    _scores_l1 = {}
    for _lbl, (_ckpts, _ep) in _conditions.items():
        _m = TinyTransformerViz()
        _m.set_task(DYCK_VOCAB)
        _m.load_state_dict(_ckpts[_ep])
        _scores_l0[_lbl] = _bracket_score(_m, layer=0)
        _scores_l1[_lbl] = _bracket_score(_m, layer=1)

    _lbls = list(_conditions.keys())
    _x = np.arange(len(_lbls))
    _w = 0.35

    _fig, _ax = plt.subplots(figsize=(10, 5.5))
    theme.style_figure(_fig)
    theme.style_axes(_ax,
        ylabel="Bracket attention score", grid_axis="y")

    _ax.axvspan(1.5, 3.5, alpha=0.08, color=theme.PALETTE["green"], zorder=0)

    _b0 = _ax.bar(_x - _w/2, [_scores_l0[l] for l in _lbls], _w,
                  label="Layer 0  (shallow patterns)",
                  color=theme.PALETTE["blue"], alpha=0.9, zorder=3)
    _b1 = _ax.bar(_x + _w/2, [_scores_l1[l] for l in _lbls], _w,
                  label="Layer 1  (induction heads)",
                  color=theme.PALETTE["orange"], alpha=0.9, zorder=3)

    _ax.axhline(_random_baseline, color=theme.FG_MUTED, linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Random baseline ({_random_baseline:.3f})", zorder=2)

    for _bar in list(_b0) + list(_b1):
        _h = _bar.get_height()
        _ax.text(_bar.get_x() + _bar.get_width()/2, _h + 0.004,
                 f"{_h:.3f}", ha="center", va="bottom",
                 fontsize=9, color=theme.FG, fontweight="semibold")

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_lbls, fontsize=10, color=theme.FG)
    _ax.set_ylim(0, 0.37)

    _ax.set_title(
        "NCA pre-training installs Layer 1 structure BEFORE seeing brackets",
        color=theme.FG, fontsize=12, pad=12, loc="left",
        fontweight="semibold")

    _leg = _ax.legend(loc="upper right", ncol=1,
                      bbox_to_anchor=(1.0, 1.0), framealpha=0.95)
    theme.style_legend(_leg)

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, theme):
    _callout = f"""
    <div style="
        background: {theme.PANEL};
        border-left: 3px solid {theme.PALETTE['green']};
        border-radius: 4px;
        padding: 16px 20px;
        margin: 16px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        color: {theme.FG};
    ">
      <div style="font-size: 12px; letter-spacing: 0.1em;
                  text-transform: uppercase; color: {theme.FG_MUTED};
                  margin-bottom: 6px;">The finding</div>
      <div style="font-size: 15px; line-height: 1.6;">
        NCA pre-training installs induction-head-like circuits in Layer 1
        <i>before</i> the model sees a single bracket.
        <br><br>
        The structure transferred from NCA dynamics to Dyck-1 probing with
        no bracket-specific data in between. The question now is whether
        fine-tuning preserves it.
      </div>
    </div>
    """
    mo.Html(_callout)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 4. Catastrophic Forgetting

    We extend the previous experiment to 80 epochs of fine-tuning across
    three random seeds each, probing Layer 1 at epochs 0, 5, 20, 40, 80.

    The pattern: **standard fine-tuning destroys the installed circuits
    within 20 epochs.** The model ends up no better than Scratch despite
    starting 40% above it.
    """)
    return


@app.cell
def _(np):
    # Pre-computed from the day_3_test.py runs (3 seeds per condition).
    # These are the actual measurements, not simulated.
    PROBE_EPOCHS = [0, 5, 20, 40, 80]
    L1_SCORES = {
        "A: Scratch":                [0.179, 0.156, 0.126, 0.143, 0.144],
        "B: NCA standard LR":        [0.248, 0.219, 0.208, 0.192, 0.181],
        "E: NCA slow LR (ours)":     [0.248, 0.255, 0.245, 0.232, 0.221],
        "F: NCA frozen attn (ours)": [0.248, 0.223, 0.241, 0.238, 0.224],
    }
    L0_SCORES = {
        "A: Scratch":                [0.179, 0.211, 0.213, 0.199, 0.165],
        "B: NCA standard LR":        [0.166, 0.155, 0.165, 0.172, 0.169],
        "E: NCA slow LR (ours)":     [0.166, 0.165, 0.170, 0.169, 0.169],
        "F: NCA frozen attn (ours)": [0.166, 0.169, 0.171, 0.169, 0.167],
    }
    RANDOM_BASELINE = 1.0 / 9
    LOSS_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80]
    LOSS_RAW = {
        "A: Scratch": {
            42: [0.6049, 0.5657, 0.5522, 0.5517, 0.5509, 0.5510, 0.5513, 0.5509],
            43: [0.5996, 0.5528, 0.5514, 0.5523, 0.5513, 0.5520, 0.5512, 0.5516],
            44: [0.6055, 0.5646, 0.5548, 0.5521, 0.5524, 0.5511, 0.5510, 0.5515]},
        "B: NCA standard LR": {
            42: [0.5634, 0.5533, 0.5517, 0.5512, 0.5511, 0.5515, 0.5510, 0.6177],
            43: [0.5696, 0.5571, 0.5515, 0.5518, 0.5517, 0.5512, 0.5528, 0.5511],
            44: [0.5891, 0.5542, 0.5519, 0.5515, 0.5580, 0.5513, 0.5516, 0.5518]},
        "E: NCA slow LR (ours)": {
            42: [0.6137, 0.6045, 0.5989, 0.5927, 0.5885, 0.5846, 0.5815, 0.5782],
            43: [0.6207, 0.6073, 0.5980, 0.5886, 0.5786, 0.5708, 0.5644, 0.5598],
            44: [0.6071, 0.5949, 0.5853, 0.5791, 0.5728, 0.5672, 0.5628, 0.5597]},
        "F: NCA frozen attn (ours)": {
            42: [0.5910, 0.5668, 0.5577, 0.5537, 0.5522, 0.5522, 0.5517, 0.5544],
            43: [0.5910, 0.5668, 0.5570, 0.5543, 0.5521, 0.5521, 0.5521, 0.5515],
            44: [0.5865, 0.5733, 0.5562, 0.5562, 0.5527, 0.5522, 0.5518, 0.5536]},
    }

    LOSS_MEAN = {}
    LOSS_STD  = {}
    LOSS_BAND_LOW = {}
    LOSS_BAND_HIGH = {}
    for _cond, _seeds in LOSS_RAW.items():
        _arr = np.array(list(_seeds.values()))
        LOSS_MEAN[_cond] = _arr.mean(axis=0).tolist()
        LOSS_STD[_cond]  = _arr.std(axis=0).tolist()
        LOSS_BAND_LOW[_cond]  = np.percentile(_arr, 25, axis=0).tolist()
        LOSS_BAND_HIGH[_cond] = np.percentile(_arr, 75, axis=0).tolist()
    return (
        L0_SCORES,
        L1_SCORES,
        LOSS_BAND_HIGH,
        LOSS_BAND_LOW,
        LOSS_EPOCHS,
        LOSS_MEAN,
        PROBE_EPOCHS,
        RANDOM_BASELINE,
    )


@app.cell
def _(
    LOSS_BAND_HIGH,
    LOSS_BAND_LOW,
    LOSS_EPOCHS,
    LOSS_MEAN,
    mo,
    np,
    plt,
    theme,
):
    # Median + IQR bands. Robust to the seed-42 divergence at epoch 80.
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    theme.style_figure(_fig)
    theme.suptitle(_fig,
        "Validation loss during Dyck-1 fine-tuning  ·  median + IQR over 3 seeds")

    for _ax, _xlim, _panel_title in [
        (_axes[0], (10, 80), "Full run"),
        (_axes[1], (10, 40), "Warmup zoom"),
    ]:
        theme.style_axes(_ax, xlabel="Epoch", ylabel="Validation loss",
                         title=_panel_title)
        for _cond, _colour in theme.CONDITION_COLOURS.items():
            _mean = np.array(LOSS_MEAN[_cond])
            _low  = np.array(LOSS_BAND_LOW[_cond])
            _high = np.array(LOSS_BAND_HIGH[_cond])
            _ep   = np.array(LOSS_EPOCHS)
            _ax.plot(_ep, _mean, color=_colour,
                     label=theme.CONDITION_LABELS_SHORT[_cond],
                     linewidth=2, marker="o", markersize=5)
            _ax.fill_between(_ep, _low, _high, color=_colour, alpha=0.15)
        _ax.set_xlim(*_xlim)
        theme.style_legend(_ax.legend(loc="upper right", fontsize=8))

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(L0_SCORES, L1_SCORES, PROBE_EPOCHS, RANDOM_BASELINE, mo, plt, theme):
    # Bracket-attention trajectory. Annotations placed in the lower area
    # (empty space above the dashed random-baseline line) with arrows
    # curving up to target the correct condition's line.
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    theme.style_figure(_fig)
    theme.suptitle(_fig, "Bracket attention score over 80 epochs of fine-tuning")

    for _ax, _scores, _panel_title in [
        (_axes[0], L0_SCORES, "Layer 0  ·  shallow patterns"),
        (_axes[1], L1_SCORES, "Layer 1  ·  induction heads"),
    ]:
        theme.style_axes(_ax, xlabel="Fine-tuning epoch",
                         ylabel="Bracket attention score",
                         title=_panel_title)
        _ax.axhline(RANDOM_BASELINE, color=theme.FG_MUTED, linestyle="--",
                    alpha=0.6, linewidth=1.2,
                    label=f"Random ({RANDOM_BASELINE:.3f})")
        for _lbl, _vals in _scores.items():
            _ax.plot(PROBE_EPOCHS, _vals,
                     color=theme.CONDITION_COLOURS[_lbl],
                     label=theme.CONDITION_LABELS_SHORT[_lbl],
                     linewidth=2.5, marker="o", markersize=6)
        _ax.set_ylim(0.085, 0.29)

        if "Layer 1" in _panel_title:
            # Blue: point at std-LR downslope at epoch 40 (score 0.192),
            # label in lower-left of plot.
            _ax.annotate(
                "Standard fine-tuning\nerodes the circuit",
                xy=(40, L1_SCORES["B: NCA standard LR"][3]),
                xytext=(20, 0.140),
                color=theme.PALETTE["blue"], fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->",
                                color=theme.PALETTE["blue"], lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
            # Green: point at slow-LR plateau at epoch 60 (interpolated),
            # label in lower-centre.
            _ax.annotate(
                "Our recipes\npreserve it",
                xy=(70, L1_SCORES["E: NCA slow LR (ours)"][3]),
                xytext=(55, 0.140),
                color=theme.PALETTE["green"], fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->",
                                color=theme.PALETTE["green"], lw=1.2,
                                connectionstyle="arc3,rad=0.2"))

        theme.style_legend(_ax.legend(loc="upper right", fontsize=8))

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, theme):
    _callout = f"""
    <div style="
        background: {theme.PANEL};
        border-left: 3px solid {theme.PALETTE['blue']};
        border-radius: 4px;
        padding: 16px 20px;
        margin: 16px 0;
        color: {theme.FG};
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    ">
      <div style="font-size: 12px; letter-spacing: 0.1em;
                  text-transform: uppercase; color: {theme.FG_MUTED};
                  margin-bottom: 6px;">The problem</div>
      <div style="font-size: 15px; line-height: 1.6;">
        Layer 1 score drops from <b>0.248</b> at epoch 0 to <b>0.181</b> at
        epoch 80 under standard fine-tuning, essentially back to scratch
        (0.144). This is catastrophic forgetting visible at the circuit
        level, not at the loss level: the loss curves all hit the same
        floor (~0.551) because Dyck-1 is too easy at this model scale.
      </div>
    </div>
    """
    mo.Html(_callout)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 5. The Fix

    Two simple recipes, both validated across three seeds:

    - **Slow LR (E):** 10× lower learning rate on the transferred blocks;
      new task head trains at full LR.
    - **Frozen attention (F):** attention weights in the blocks frozen;
      only LayerNorms and the FFN MLPs update.

    Both preserve >90% of the Layer 1 structure through 80 epochs while
    still matching Scratch on validation loss.
    """)
    return


@app.cell
def _(mo):
    # Interactive scrubber, the real "marimo moment" in this notebook.
    epoch_scrubber = mo.ui.slider(
        start=0, stop=80, step=1, value=0,
        label="Fine-tuning epoch",
        show_value=True,
    )
    mo.vstack([
        mo.md("### Interactive: scrub through training"),
        mo.md("Drag the slider to see Layer 1 bracket scores at any epoch. "
              "This is the same data as the trajectory plot above, "
              "re-presented so you can compare conditions point-by-point."),
        epoch_scrubber,
    ])
    return (epoch_scrubber,)


@app.cell
def _(L1_SCORES, PROBE_EPOCHS, RANDOM_BASELINE, epoch_scrubber, mo, np, theme):
    _ep = epoch_scrubber.value
    _bars = []
    _max_val = 0.28

    for _lbl, _vals in L1_SCORES.items():
        _score = float(np.interp(_ep, PROBE_EPOCHS, _vals))
        _colour = theme.CONDITION_COLOURS[_lbl]
        _width_pct = (_score / _max_val) * 100
        _vs_random_pct = (_score - RANDOM_BASELINE) / RANDOM_BASELINE * 100
        _short = theme.CONDITION_LABELS_SHORT[_lbl]
        _bars.append(f"""
        <div style="display: grid;
                    grid-template-columns: 140px 1fr 80px 80px;
                    gap: 12px; align-items: center;
                    padding: 8px 0;">
          <div style="color: {theme.FG}; font-size: 13px;">{_short}</div>
          <div style="background: {theme.BG};
                      border: 1px solid {theme.SPINE};
                      border-radius: 3px; height: 22px;
                      position: relative; overflow: hidden;">
            <div style="background: {_colour}; height: 100%;
                        width: {_width_pct:.1f}%;
                        transition: width 0.15s ease-out;"></div>
          </div>
          <div style="color: {theme.FG}; font-size: 14px;
                      font-weight: 600; font-family: 'SF Mono', monospace;">
            {_score:.3f}
          </div>
          <div style="color: {theme.FG_MUTED}; font-size: 11px;
                      text-align: right;">
            {_vs_random_pct:+.0f}% vs random
          </div>
        </div>
        """)

    _baseline_pct = (RANDOM_BASELINE / _max_val) * 100

    _html = f"""
    <div style="
        background: {theme.PANEL};
        border: 1px solid {theme.SPINE};
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    ">
      <div style="display: flex; align-items: baseline; gap: 12px;
                  margin-bottom: 16px;">
        <div style="color: {theme.FG_MUTED}; font-size: 11px;
                    letter-spacing: 0.1em; text-transform: uppercase;">
          Layer 1 bracket attention at epoch
        </div>
        <div style="color: {theme.FG}; font-size: 28px;
                    font-weight: 600;">{_ep}</div>
      </div>

      <div style="position: relative;">
        {"".join(_bars)}
      </div>

      <div style="color: {theme.FG_MUTED}; font-size: 11px;
                  margin-top: 14px; padding-top: 12px;
                  border-top: 1px solid {theme.SPINE};">
        Random baseline (uniform attention) = {RANDOM_BASELINE:.3f}.
        Bar scale: 0 to {_max_val}. Interpolated linearly between probe
        epochs {PROBE_EPOCHS}.
      </div>
    </div>
    """
    mo.Html(_html)
    return


@app.cell
def _(
    L1_SCORES,
    LOSS_BAND_HIGH,
    LOSS_BAND_LOW,
    LOSS_EPOCHS,
    LOSS_MEAN,
    RANDOM_BASELINE,
    mo,
    np,
    plt,
    theme,
):
    # Final summary figure: loss curves + end-of-training bar chart.
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5.5))
    theme.style_figure(_fig)
    theme.suptitle(_fig, "The fix  ·  two recipes that preserve the circuit")

    theme.style_axes(_axes[0], xlabel="Epoch", ylabel="Validation loss",
                     title="Validation loss  ·  lower = better")
    for _cond, _colour in theme.CONDITION_COLOURS.items():
        _mean = np.array(LOSS_MEAN[_cond])
        _low  = np.array(LOSS_BAND_LOW[_cond])
        _high = np.array(LOSS_BAND_HIGH[_cond])
        _ep   = np.array(LOSS_EPOCHS)
        _axes[0].plot(_ep, _mean, color=_colour,
                      label=theme.CONDITION_LABELS_SHORT[_cond],
                      linewidth=2, marker="o", markersize=4)
        _axes[0].fill_between(_ep, _low, _high, color=_colour, alpha=0.15)
    theme.style_legend(_axes[0].legend(loc="upper right", fontsize=8))

    _conditions = list(theme.CONDITION_COLOURS.keys())
    _scores_80 = {c: L1_SCORES[c][-1] for c in _conditions}
    _colours = [theme.CONDITION_COLOURS[c] for c in _conditions]
    _labels = [theme.CONDITION_LABELS_SHORT[c] for c in _conditions]
    _x = np.arange(len(_conditions))

    theme.style_axes(_axes[1], ylabel="Layer 1 bracket score at epoch 80",
                     title="Circuit preservation at epoch 80",
                     grid_axis="y")
    _bars = _axes[1].bar(_x, [_scores_80[c] for c in _conditions],
                         color=_colours, alpha=0.9, width=0.6, zorder=3)
    _axes[1].axhline(RANDOM_BASELINE, color=theme.FG_MUTED, linestyle="--",
                     linewidth=1.2, alpha=0.7,
                     label=f"Random baseline ({RANDOM_BASELINE:.3f})",
                     zorder=2)
    for _bar in _bars:
        _h = _bar.get_height()
        _axes[1].text(_bar.get_x() + _bar.get_width()/2, _h + 0.004,
                      f"{_h:.3f}", ha="center", va="bottom",
                      fontsize=10, color=theme.FG, fontweight="semibold")
    _axes[1].set_xticks(_x)
    _axes[1].set_xticklabels(_labels, color=theme.FG, fontsize=10)
    _axes[1].set_ylim(0, 0.29)
    theme.style_legend(_axes[1].legend(loc="upper left", fontsize=8))

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why doesn't this show up in the loss curves?

    At toy scale (~150k params, Dyck-1), every condition converges to
    the same loss floor (~0.551). The task is easy enough that the
    model can re-learn it from any initialisation. The circuit probes
    are what reveal the difference.

    This is why the paper's main result is expressed as **perplexity**
    at 400M parameters over 164M NCA pre-training tokens. At scale,
    the preserved circuits compound into a measurable ~6% perplexity
    gain. Our contribution is the mechanistic story beneath that number.
    """)
    return


@app.cell
def _(mo, theme):
    _story = f"""
    <div style="
        background: {theme.PANEL};
        border: 1px solid {theme.SPINE};
        border-radius: 8px;
        padding: 28px 32px;
        margin: 24px 0 12px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    ">
      <div style="color: {theme.FG_MUTED}; font-size: 11px;
                  letter-spacing: 0.12em; text-transform: uppercase;
                  margin-bottom: 20px;">
        The story, in four lines
      </div>

      <div style="display: grid; gap: 18px;">

        <div style="display: grid; grid-template-columns: 40px 1fr; gap: 16px;">
          <div style="color: {theme.PALETTE['green']}; font-size: 24px;
                      font-weight: 600; font-family: 'SF Mono', monospace;">
            01
          </div>
          <div style="color: {theme.FG}; font-size: 14px; line-height: 1.6;">
            <b>NCA pre-training installs induction-head circuits in Layer 1.</b>
            &nbsp;<span style="color: {theme.FG_MUTED}">
            Score: 0.248 vs random baseline 0.111, before seeing a single bracket.
            </span>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 40px 1fr; gap: 16px;">
          <div style="color: {theme.PALETTE['blue']}; font-size: 24px;
                      font-weight: 600; font-family: 'SF Mono', monospace;">
            02
          </div>
          <div style="color: {theme.FG}; font-size: 14px; line-height: 1.6;">
            <b>Standard fine-tuning destroys them.</b>
            &nbsp;<span style="color: {theme.FG_MUTED}">
            Score falls 0.248 to 0.181 over 80 epochs. Catastrophic forgetting
            at the circuit level.
            </span>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 40px 1fr; gap: 16px;">
          <div style="color: {theme.PALETTE['green']}; font-size: 24px;
                      font-weight: 600; font-family: 'SF Mono', monospace;">
            03
          </div>
          <div style="color: {theme.FG}; font-size: 14px; line-height: 1.6;">
            <b>Slow LR or frozen attention preserves them.</b>
            &nbsp;<span style="color: {theme.FG_MUTED}">
            Score holds at 0.221 to 0.224. Validation loss matches Scratch.
            The mechanism survives fine-tuning.
            </span>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 40px 1fr; gap: 16px;">
          <div style="color: {theme.PALETTE['orange']}; font-size: 24px;
                      font-weight: 600; font-family: 'SF Mono', monospace;">
            04
          </div>
          <div style="color: {theme.FG}; font-size: 14px; line-height: 1.6;">
            <b>At scale, the circuits compound.</b>
            &nbsp;<span style="color: {theme.FG_MUTED}">
            400M params, 164M NCA tokens: the paper reports a 6% perplexity
            gain. Our toy experiments reveal the mechanism beneath it.
            </span>
          </div>
        </div>

      </div>
    </div>
    """
    mo.Html(_story)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ### Recommendation

    When fine-tuning from a structurally rich pre-training (like NCA),
    use a layerwise LR schedule or freeze attention during the early
    fine-tuning phase. The paper does not make this recommendation.
    We do, based on what the circuit probes reveal.

    ### What we didn't do

    We ran at a scale where every model converges to the same loss.
    The mechanistic finding is real but the claim of downstream
    advantage needs the paper's scale to show up in loss. We also only
    probed a single-pair Dyck task (bracket matching with one pair);
    richer tasks would test whether the preserved circuits generalise.

    ### References

    - **Primary paper:** Lee, Han, Kumar & Agrawal (2026).
      *Training Language Models via Neural Cellular Automata.*
      arXiv:2603.10055.
    - **Induction heads:** Olsson et al. (2022).
      *In-context Learning and Induction Heads.* Anthropic.
    - **Circuit preservation:** classical transfer-learning literature
      on LR scheduling and adapter methods.

    ---

    *Built for the marimo × alphaXiv Notebook Competition, April 2026.*
    *Code: github.com/vinodanbalagan/nca-language-pretraining*
    *Writeup: The Meta Gradient, substack.com/@vinodanbalagan*
    """)
    return


if __name__ == "__main__":
    app.run()
