"""
theme.py
--------
Single source of visual truth for the notebook.

Why this file exists:
- Every figure should look like it came from the same notebook, not 12 different ones.
- We want a consistent palette for conditions (Scratch vs NCA vs ours) so readers
  build muscle memory across figures.
- Matplotlib defaults on a dark background look amateur without deliberate choices
  about spines, ticks, grid, and font weights.

How to use:
    from theme import BG, FG, PALETTE, CONDITION_COLOURS, style_axes, style_figure

    fig, ax = plt.subplots(figsize=(10, 5))
    style_figure(fig)
    # ... plot stuff ...
    style_axes(ax, title="...", xlabel="...", ylabel="...")
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Core colours
# ---------------------------------------------------------------------------
# Background is slightly lifted off pure black — pure black eats detail in
# matplotlib rasters. #0f1419 reads as "dark" but preserves anti-aliasing.
BG = "#0f1419"           # figure + axes background
PANEL = "#151a21"        # slightly lighter for nested panels / callouts
FG = "#e6e6e6"           # primary text — off-white, not pure white (less harsh)
FG_MUTED = "#8a94a6"     # secondary text (axis labels, captions)
GRID = "#2a303a"         # grid lines — visible but not shouty
SPINE = "#3a4250"        # axis spines

# ---------------------------------------------------------------------------
# Condition palette
# ---------------------------------------------------------------------------
# Semantic colours — these map to experimental conditions and are used
# consistently across every figure. The reader should learn: green = "our fix".
CONDITION_COLOURS = {
    "A: Scratch":                "#8a94a6",   # muted grey — the baseline
    "B: NCA standard LR":        "#4dabf7",   # blue — the paper's recipe
    "E: NCA slow LR (ours)":     "#51cf66",   # green — our contribution
    "F: NCA frozen attn (ours)": "#ffa94d",   # orange — our contribution
}

# Short-label version for plots where the full name won't fit.
CONDITION_LABELS_SHORT = {
    "A: Scratch":                "Scratch",
    "B: NCA standard LR":        "NCA (std LR)",
    "E: NCA slow LR (ours)":     "NCA + slow LR",
    "F: NCA frozen attn (ours)": "NCA + frozen attn",
}

# ---------------------------------------------------------------------------
# Accent palette (for one-off highlights)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":   "#4dabf7",
    "green":  "#51cf66",
    "orange": "#ffa94d",
    "red":    "#ff6b6b",
    "purple": "#b197fc",
    "yellow": "#ffd43b",
    "cyan":   "#3bc9db",
}

# Two-colour categorical map for n=2 NCA grids.
# These are chosen to have good contrast on the dark background AND to
# photocopy/grayscale reasonably. Not red/green (colourblind).
NCA_BINARY = ["#1e3a5f", "#ffa94d"]  # deep blue + warm orange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def style_figure(fig):
    """Apply background + suptitle colour to a figure."""
    fig.patch.set_facecolor(BG)
    # Suptitle colour needs to be set after suptitle is added, so we don't
    # touch it here — callers handle it via style_axes or explicitly.
    return fig


def style_axes(
    ax,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    grid=True,
    grid_axis="both",
    show_top_right_spines=False,
):
    """
    Apply the house style to a single axes.

    Keyword-only args to prevent order-confusion bugs.
    """
    ax.set_facecolor(BG)

    if title is not None:
        ax.set_title(title, color=FG, fontsize=12, pad=10, loc="left",
                     fontweight="semibold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=FG_MUTED, fontsize=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=FG_MUTED, fontsize=10)

    ax.tick_params(colors=FG_MUTED, labelsize=9)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(show_top_right_spines)
        if show_top_right_spines:
            ax.spines[side].set_color(SPINE)
            ax.spines[side].set_linewidth(0.8)

    if grid:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.6,
                zorder=0)
        ax.set_axisbelow(True)

    return ax


def style_legend(legend):
    """Apply house style to a matplotlib legend."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(PANEL)
    frame.set_edgecolor(SPINE)
    frame.set_linewidth(0.8)
    frame.set_alpha(0.95)
    for text in legend.get_texts():
        text.set_color(FG)
        text.set_fontsize(9)
    return legend


def style_image_axes(ax, title=None):
    """
    For imshow panels (NCA grids, attention heatmaps) where we don't want
    spines or ticks — just a title and a clean frame.
    """
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    if title is not None:
        ax.set_title(title, color=FG, fontsize=11, pad=8)
    return ax


def suptitle(fig, text, y=0.98):
    """Apply a suptitle with house style."""
    fig.suptitle(text, color=FG, fontsize=13, fontweight="semibold", y=y)


# ---------------------------------------------------------------------------
# Global matplotlib rcParams
# ---------------------------------------------------------------------------
def apply_rc():
    """
    Call once at the top of the notebook to set global matplotlib defaults.
    This handles things that are tedious to set per-figure (font family,
    default figure DPI, default line width).
    """
    mpl.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "savefig.facecolor": BG,
        "savefig.dpi":       120,
        "figure.dpi":        110,

        # Typography — use the system sans stack, fall back to DejaVu.
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "SF Pro Display", "Helvetica Neue",
                              "Arial", "DejaVu Sans"],
        "font.size":         10,

        # Line defaults
        "lines.linewidth":   2.0,
        "lines.markersize":  5,

        # Text
        "text.color":        FG,
        "axes.labelcolor":   FG_MUTED,
        "xtick.color":       FG_MUTED,
        "ytick.color":       FG_MUTED,
        "axes.edgecolor":    SPINE,

        # Legend
        "legend.frameon":    True,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  SPINE,
        "legend.fontsize":   9,

        # Tighter default padding
        "axes.titlepad":     10,
        "axes.labelpad":     6,
    })
