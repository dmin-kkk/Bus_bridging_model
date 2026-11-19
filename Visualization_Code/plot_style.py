from __future__ import annotations

import matplotlib.pyplot as plt


FONT_FAMILY = "Times New Roman"
FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]
BASE_FONT_SIZE = 15
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 15
LEGEND_FONT_SIZE = 13
SAVEFIG_DPI = 300
SAVEFIG_BBOX = "tight"
GRID_COLOR = "#C7CCD1"
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.8
LEGEND_EDGE_COLOR = "#BFBFBF"
DEFAULT_FIGSIZE = (10, 6)


def apply_global_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_SERIF,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "savefig.dpi": SAVEFIG_DPI,
            "savefig.bbox": SAVEFIG_BBOX,
        }
    )


def apply_axes_style(ax, xlabel: str, ylabel: str, xlim=None, ylim=None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        color=GRID_COLOR,
        alpha=0.9,
    )
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor=LEGEND_EDGE_COLOR)
