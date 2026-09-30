"""Common matplotlib style: one palette, one colour per model everywhere."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8983"

# categorical palette (fixed order, colour-vision-deficiency validated)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}

# colour follows the model, in every figure
MODEL_COLORS = {
    "Persistence": NEUTRAL,
    "RW+drift": "#b5b4ad",
    "Linear trend (LS)": "#6f6e69",
    "SES": "#a09f98",
    "AR(4)-OLS": VIOLET,
    "VRNN-AR": ORANGE,
    "VRNN-Direct": YELLOW,
    "Seq2Seq-Attn": AQUA,
    "Transformer": BLUE,
}


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def model_color(name: str) -> str:
    base = name.split(" [")[0]
    return MODEL_COLORS.get(base, INK2)
