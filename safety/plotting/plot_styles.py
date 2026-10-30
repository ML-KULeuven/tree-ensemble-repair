"""Figure style and palette shared by the plotting scripts in this folder.

Every figure uses the same style (call `set_style()` at the top of each plotting
function, since seaborn and rcParams are global) and draws a method in the same
colour wherever it appears. The assignment mirrors the fairness-repair figures,
so the two papers' plots read the same way:

  MILP Repair (composite)   green
  Original (unrepaired)     dark grey        (the "input model" slot)
  baselines                 softened warm colours, one fixed each
                            (MH red, FALoss orange, PT dark yellow)
  ablated / variant MILPs   one fixed colour each (MILP-L1 mid grey,
                            lambda sweep = ramp of the MILP green)
  run outcomes              softened (timeout pink, infeasible purple,
                            OOM blue-grey, max-iterations brown)
  plots of our MILP alone   colour the metric instead (blue accuracy/%Safe,
                            dark grey runtime)
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# ── Palette ──────────────────────────────────────────────────────────────────

def _soften(color, strength):
    """`color` blended with white: what it looks like at alpha=`strength` on a
    white background, but as an opaque colour (no see-through grid or overlaps)."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex([strength * c + (1 - strength) for c in (r, g, b)])


def ramp(color, n, lightest=0.35, darkest=1.0):
    """`n` steps of one hue from light to full strength (sequential encoding,
    e.g. increasing lambda). `lightest`/`darkest` are blend strengths with white."""
    if n == 1:
        return [mcolors.to_hex(color)]
    return [_soften(color, lightest + (darkest - lightest) * i / (n - 1)) for i in range(n)]


# methods
C_MILP     = "#55A868"  # green       - MILP Repair, composite (hinge + L1) objective
C_ORIGINAL = "#4D4D4D"  # dark grey   - the unrepaired policy (same slot as LOP)

# baselines: softened so they stay lighter than our own method
BASELINE_STRENGTH = 0.75                        # 1 = full colour, 0 = white
C_MH     = _soften("#C20D13", BASELINE_STRENGTH)   # soft red         - model hardening (retraining-augment)
C_FALOSS = _soften("#DD5709", BASELINE_STRENGTH)   # soft orange      - fault-aware loss (retraining-loss)
C_PT     = _soften("#EEBC07", BASELINE_STRENGTH)   # soft dark yellow - penalty trees

# ablated / variant MILPs
C_L1       = "#9E9E9E"  # mid grey    - pure L1 objective (milp-l1)
C_ACC_ONLY = "#4D4D4D"  # dark grey   - composite with lambda = 0
C_BUDGET   = "#4C72B0"  # blue        - hard accuracy budget variant (if used)


def lambda_ramp(n):
    """Colours for an increasing-lambda sweep of the MILP: light -> full green."""
    return ramp(C_MILP, n)


# run outcomes: failures, softened like the baselines
OUTCOME_STRENGTH = 0.75                         # 1 = full colour, 0 = white
C_TIMEOUT    = _soften("#D9469B", OUTCOME_STRENGTH)  # soft pink      - solver / verifier timed out
C_INFEASIBLE = _soften("#6A3D9A", OUTCOME_STRENGTH)  # soft purple    - MILP infeasible
C_OOM        = _soften("#4C72B0", OUTCOME_STRENGTH)  # soft blue      - out of memory
C_MAXIT      = _soften("#8C564B", OUTCOME_STRENGTH)  # soft brown     - iteration limit reached

# metrics, for plots that show our MILP alone
C_ACC  = "#4C72B0"      # blue        - accuracy-type quantity (%Safe, %Goal, fidelity)
C_RT   = "#4D4D4D"      # dark grey   - runtime
C_DOT  = "#4C72B0"      # blue        - one point per benchmark / fold (scatters)
C_MEAN = "#1A3F7A"      # dark blue   - mean marker

# annotations
C_REF      = "gray"     # grey        - reference lines (no change, 100%)
C_SUBTITLE = "#444444"  # dark grey   - secondary caption under a title
# black - boxplot outlines, whiskers and medians; pass it as `linecolor`, or
# seaborn derives a grey from the darkest fill, which differs per figure
C_BOX_LINE = "black"

# method label -> colour, for scripts that build columns from a method list
METHOD_COLORS = {
    "Original": C_ORIGINAL,
    "MILP": C_MILP,
    "MILP-L1": C_L1,
    "MH": C_MH,
    "FALoss": C_FALOSS,
    "PT": C_PT,
}

# ── Style ────────────────────────────────────────────────────────────────────

# Base sizes (points) at scale 1: the sizes of the earlier raincloud figures
# (14.5in canvas). set_style(scale) multiplies them if ever needed.
BASE_SIZES = {                  # sizes of the earlier raincloud figures
    "font.size":        14,
    "axes.titlesize":   18,
    "axes.labelsize":   17,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  14,
}
_ANNOT_BASE    = 13     # value labels on bars, NA counts
_SUBTITLE_BASE = 14     # caption under a title, row labels of annotation strips
_DELTA_BASE    = 16     # headline numbers under an axis (e.g. Δ vs Original)
FONT_SCALE = 1.0        # current scale, set by set_style()
ANNOT_SIZE = _ANNOT_BASE
SUBTITLE_SIZE = _SUBTITLE_BASE
DELTA_SIZE = _DELTA_BASE
DPI = 300


def set_style(scale=1.0):
    """The one figure style: sans-serif, large type, bold titles.
    `scale` multiplies every font size (rcParams and the annotation sizes
    exported by this module), so a whole figure can be made larger or smaller."""
    global FONT_SCALE, ANNOT_SIZE, SUBTITLE_SIZE, DELTA_SIZE
    import seaborn as sns
    sns.set_theme(style="white")    # also resets font.family to sans-serif
    FONT_SCALE = float(scale)
    ANNOT_SIZE = _ANNOT_BASE * FONT_SCALE
    SUBTITLE_SIZE = _SUBTITLE_BASE * FONT_SCALE
    DELTA_SIZE = _DELTA_BASE * FONT_SCALE
    plt.rcParams.update({k: v * FONT_SCALE for k, v in BASE_SIZES.items()})
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.titlepad":    12,
        "pdf.fonttype":     42,     # editable text in PDFs (TrueType, not Type 3)
        "ps.fonttype":      42,
    })


def style_axes(ax, grid="both"):
    """Visible 0.8 spines, white background, light dotted grid behind the data.

    `grid="y"` for categorical x axes (bars, boxes, rainclouds), where vertical
    lines would only cut through the marks.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.set_facecolor("white")
    ax.grid(axis=grid, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def save(fig, stem, tight=True):
    """Write {stem}.pdf and {stem}.png at DPI; returns the two paths.

    `tight=False` for panels placed side by side in the paper: a tight bbox
    crops each figure to its own content, so same-sized canvases would come out
    at different widths and scale to different font sizes.
    """
    paths = []
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight" if tight else None)
        paths.append(path)
    plt.close(fig)
    return paths
