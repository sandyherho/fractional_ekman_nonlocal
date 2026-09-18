"""Figure style, palette, and multi-format export.

Conventions follow the usual physics-journal layout.  Panel labels sit outside
the axes frame, above its top-left corner.  Legends sit below the axes as a
single unframed figure-level legend, so that no curve is ever obscured and no
explanatory text competes with the data.  Figures carry no titles and no
in-panel annotation: numerical values belong in the reports and the manuscript
captions, not inside the axes.

Every plotted quantity is dimensionless.  Depths are in units of the
generalized Ekman depth delta_gamma, times in units of 1/f, speeds relative to
the surface speed, and stresses relative to tau.  The one exception is the
Caputo degeneracy panel, whose abscissa is in the arbitrary units of a
synthetic test profile and where only the slope carries meaning.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("Agg")

__all__ = ["OKABE_ITO", "GAMMA_COLORS", "setup", "panel_label",
           "outside_legend", "gamma_handles", "save"]

# Okabe and Ito colorblind-safe qualitative palette
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#8C8C8C",
}

# fixed assignment of gamma to colour, used consistently across every figure
GAMMA_COLORS = [
    OKABE_ITO["purple"], OKABE_ITO["vermil"], OKABE_ITO["orange"],
    OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["black"],
]

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(_HERE), "outputs", "figures")


def setup():
    """Apply the figure style used by every script in this repository."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, text, dx=0.0, dy=1.03):
    """Panel label outside the axes, above its top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def gamma_handles(gammas, colors, styles=None):
    """Legend proxies for a family of gamma values."""
    styles = styles or ["-"] * len(gammas)
    return ([Line2D([], [], color=c, ls=s, lw=1.3)
             for c, s in zip(colors, styles)],
            [rf"$\gamma={g:g}$" for g in gammas])


def outside_legend(fig, handles, labels, ncol=None, y=0.0):
    """Single unframed figure-level legend below the axes."""
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                      borderaxespad=0.0)


def save(fig, stem, dpi=600):
    """Write a vector PDF and a high-resolution PNG.

    The PDF is the submission artifact; fonts are embedded as Type 42 so that
    text remains selectable and editable.  The PNG is rendered at ``dpi`` for
    screen use and for preview in the repository.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths[0]
