"""Figure style and palette shared by every collect_*.py script.

Every figure uses the same style (call `set_style()` at the top of each plotting
function, since seaborn and rcParams are global) and draws a method in the same
colour wherever it appears:

  MILP Repair (composite)   green
  LOP                       dark grey
  baselines                 softened warm colours, one fixed each (FATT red, BuDRO orange)
  ablated / variant MILPs   one fixed colour each (budget blue, lambda=0 dark grey, L1 mid grey)
  run outcomes              softened (timeout pink, infeasible purple)
  plots of our MILP alone   colour the metric instead (blue accuracy, dark grey runtime)
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

# ── Palette ──────────────────────────────────────────────────────────────────

def _soften(color, strength):
    """`color` blended with white: what it looks like at alpha=`strength` on a
    white background, but as an opaque colour (no see-through grid or overlaps)."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex([strength * c + (1 - strength) for c in (r, g, b)])


# methods
C_MILP = "#55A868"      # green       - MILP Repair, composite objective
C_LOP  = "#4D4D4D"      # dark grey   - LOP-compressed model

# baselines: softened so they stay lighter than our own method;
# FairGBM keeps its slot for when it comes back
BASELINE_STRENGTH = 0.75                        # 1 = full colour, 0 = white
C_FATT    = _soften("#C20D13", BASELINE_STRENGTH)   # soft red         - FATT
C_BUDRO   = _soften("#DD5709", BASELINE_STRENGTH)   # soft orange      - BuDRO
C_FAIRGBM = _soften("#EEBC07", BASELINE_STRENGTH)   # soft dark yellow - FairGBM (currently unused)

# ablated / variant MILPs
C_BUDGET   = "#4C72B0"  # blue        - hard balanced-accuracy budget (milp-b)
C_ACC_ONLY = "#4D4D4D"  # dark grey   - composite with lambda = 0 (ablation 2)
C_L1       = "#9E9E9E"  # mid grey    - pure L1 objective (ablation 2)

# run outcomes (budget outcome figure): failures, softened like the baselines
OUTCOME_STRENGTH = 0.75                         # 1 = full colour, 0 = white
C_TIMEOUT    = _soften("#D9469B", OUTCOME_STRENGTH)  # soft pink   - budget MILP timed out
C_INFEASIBLE = _soften("#6A3D9A", OUTCOME_STRENGTH)  # soft purple - budget MILP infeasible

# metrics, for plots that show our MILP alone
C_ACC  = "#4C72B0"      # blue        - accuracy-type quantity
C_RT   = "#4D4D4D"      # dark grey   - runtime
C_DOT  = "#4C72B0"      # blue        - one point per fold (runtime scatters)
C_MEAN = "#1A3F7A"      # dark blue   - per-dataset mean marker

# annotations
C_REF      = "gray"     # grey        - reference lines (no change, 100%)
C_SUBTITLE = "#444444"  # dark grey   - secondary caption under a title
# black - boxplot outlines, whiskers and medians; pass it as `linecolor`, or
# seaborn derives a grey from the darkest fill, which differs per figure
C_BOX_LINE = "black"

# ── Style ────────────────────────────────────────────────────────────────────

ANNOT_SIZE    = 13      # value labels on bars
SUBTITLE_SIZE = 14      # caption under a title (e.g. Spearman rho)
DPI = 300


def set_style():
    """The one figure style: sans-serif, large type, bold titles."""
    sns.set_theme(style="white")    # also resets font.family to sans-serif
    plt.rcParams.update({
        "font.size":        17,
        "axes.titlesize":   17,
        "axes.titleweight": "bold",
        "axes.titlepad":    12,
        "axes.labelsize":   18,
        "xtick.labelsize":  15,
        "ytick.labelsize":  15,
    })


def style_axes(ax, grid="both"):
    """Visible 0.8 spines, white background, light dotted grid behind the data.

    `grid="y"` for categorical x axes (bars, boxes), where vertical lines
    would only cut through the boxes.
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
