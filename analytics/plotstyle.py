"""One shared figure style: font size, colour cycle, 150 dpi PNG under reports/fig/ via save()."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import settings  # noqa: E402

FIG_DIR = settings.HERE / "reports" / "fig"
DPI = 150
FONT_SIZE = 9
# Categorical slots in fixed order (reference light-mode palette; direct labels or a legend always
# accompany them, so identity is never colour alone).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"
MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply() -> None:
    matplotlib.rcParams.update({
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 1, "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e3e2dc", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 6, "figure.dpi": DPI, "savefig.dpi": DPI,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def subplots(nrows=1, ncols=1, size=(6.4, 3.8), **kw):
    apply()
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained", **kw)


def save(fig, name: str, outdir=None) -> Path:
    """Write `<outdir or reports/fig>/<name>.png` at 150 dpi, close the figure, return the path."""
    out = Path(outdir) if outdir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    return path


apply()
