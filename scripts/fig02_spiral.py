"""Figure 2. Hodographs of the nonlocal Ekman spiral.

Each panel traces psi(zeta) in the velocity plane from the surface to depth,
with the radius compressed as |psi|^(1/3) so that the strongly decaying
classical case stays legible.  The dotted radial line is the surface deflection
and the dashed radial line the deep asymptote.  Angles are tabulated in
outputs/reports/closed_forms.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bayu.core import Params, deflection_angle, asymptotic_angle
from bayu.solution import velocity
from bayu.plotting import (setup, OKABE_ITO as C, GAMMA_COLORS, panel_label,
                           outside_legend, save)
from bayu.io_utils import write_csv

GAMS = [0.25, 0.50, 0.75, 1.00]
COLS = [GAMMA_COLORS[0], GAMMA_COLORS[1], GAMMA_COLORS[3], C["black"]]

setup()
fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.4))
out = {}
z = np.concatenate([[0.0], np.logspace(-3.5, 2.0, 2600)])

for ax, g, col, lab in zip(axes.ravel(), GAMS, COLS, "abcd"):
    prm = Params(gamma=g)
    ps = velocity(prm, z)
    r = (np.abs(ps) / abs(ps[0])) ** (1.0 / 3.0)
    th = np.angle(ps) - np.angle(prm.tau)
    x, y = r * np.cos(th), r * np.sin(th)
    out[f"g{g}_x"], out[f"g{g}_y"] = x, y

    ax.set_aspect("equal")
    ax.axhline(0, color=C["grey"], lw=0.5, zorder=0)
    ax.axvline(0, color=C["grey"], lw=0.5, zorder=0)
    a0 = np.radians(deflection_angle(g))
    ax.plot([0, 1.10 * np.cos(a0)], [0, 1.10 * np.sin(a0)], color=col,
            lw=0.7, ls=":", zorder=1)
    if g < 1.0:
        ai = np.radians(asymptotic_angle(g))
        ax.plot([0, 1.10 * np.cos(ai)], [0, 1.10 * np.sin(ai)], color=col,
                lw=0.7, ls="--", zorder=1)
    ax.plot(x, y, color=col, lw=1.4, zorder=2)
    ax.plot(x[0], y[0], "o", ms=4.5, color=col, zorder=3)
    ax.annotate("", xy=(1.14, 0.0), xytext=(0.72, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=C["vermil"], lw=1.1))
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    panel_label(ax, rf"({lab})  $\gamma={g:.2f}$")

for ax in axes[1, :]:
    ax.set_xlabel(r"$(|\psi|/|\psi_0|)^{1/3}\cos\theta$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$(|\psi|/|\psi_0|)^{1/3}\sin\theta$")

fig.tight_layout()
handles = [
    Line2D([], [], color=C["grey"], lw=1.4),
    Line2D([], [], color=C["grey"], lw=0.7, ls=":"),
    Line2D([], [], color=C["grey"], lw=0.7, ls="--"),
    Line2D([], [], color=C["vermil"], lw=1.1),
]
labels = [r"$\psi(\zeta)$", r"$\arg\psi(0)$", r"$\arg\psi(\infty)$",
          r"$\tau$"]
outside_legend(fig, handles, labels, ncol=4, y=0.0)
print(save(fig, "fig02_spiral"))
out["zeta"] = z
write_csv("fig02_hodographs", out)
