"""Figure 0. Schematic of the nonlocal closure.

Single panel.  The classical Fickian stress at depth zeta samples the shear at
that depth alone.  The Scott-Blair stress weights the shear over the entire
column above the evaluation depth by the kernel (zeta - xi)^-gamma,
shown as the filled profile.  The key is part of the diagram, and no
numerical values appear.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from bayu.plotting import setup, OKABE_ITO as C, save

setup()
fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.set_xlim(-1.30, 0.95)
ax.set_ylim(1.06, -0.26)
ax.axis("off")

ZS, GAM = 0.60, 0.6

# water column and free surface
ax.fill_between([-1.30, 0.95], 0.0, 1.06, color="#F2F6F8", zorder=0)
ax.plot([-1.30, 0.95], [0, 0], color=C["black"], lw=1.6, zorder=4)
ax.plot([0, 0], [0.0, 1.06], color=C["black"], lw=1.0, zorder=4)

# wind stress
ax.annotate("", xy=(0.62, -0.13), xytext=(-0.32, -0.13),
            arrowprops=dict(arrowstyle="-|>", color=C["vermil"], lw=1.7,
                            mutation_scale=11))
ax.text(0.68, -0.13, r"$\tau$", color=C["vermil"], fontsize=11, va="center")
ax.text(0.92, 0.055, r"$\zeta=0$", fontsize=8, ha="right", color=C["black"])

# depth axis
ax.annotate("", xy=(0.86, 0.40), xytext=(0.86, 0.14),
            arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=0.9))
ax.text(0.80, 0.27, r"$\zeta$", color=C["grey"], fontsize=9, ha="right",
        va="center")

# nonlocal kernel weight over the column above the evaluation depth
xi = np.linspace(0.0, ZS - 1e-4, 500)
w = (ZS - xi) ** (-GAM)
w = 0.36 * w / w[0]
w = np.minimum(w, 1.02)
ax.fill_betweenx(xi, -w, 0.0, color=C["skyblue"], alpha=0.60, lw=0, zorder=1)
ax.plot(-w, xi, color=C["skyblue"], lw=1.1, zorder=2)
ax.plot([-w[-1], 0.0], [ZS, ZS], color=C["skyblue"], lw=1.1, zorder=2)

ax.text(-1.28, 0.30, r"$(\zeta-\xi)^{-\gamma}$", color=C["blue"], fontsize=9,
        va="center", ha="left")
ax.annotate("", xy=(-0.42, 0.30), xytext=(-0.72, 0.30),
            arrowprops=dict(arrowstyle="-|>", color=C["skyblue"], lw=0.9))
ax.text(-0.09, 0.10, r"$\xi$", color=C["blue"], fontsize=9, ha="right",
        va="center")

# evaluation depth
ax.plot([0], [ZS], marker="o", ms=6, color=C["blue"], zorder=6)
ax.text(0.06, ZS + 0.075, r"$T(\zeta)$", color=C["blue"], fontsize=9,
        va="center")

# local comparison
ax.plot([0.44, 0.44], [ZS - 0.016, ZS + 0.016], color=C["grey"], lw=5.5,
        solid_capstyle="butt", zorder=5)
ax.annotate("", xy=(0.06, ZS), xytext=(0.40, ZS),
            arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=0.9))

ax.text(-1.28, 1.00, "nonlocal", color=C["blue"], fontsize=9, ha="left")
ax.text(0.92, 0.92, "local", color=C["grey"], fontsize=9, ha="right")
ax.text(0.92, 1.00, r"$\gamma=1$", color=C["grey"], fontsize=8, ha="right")

print(save(fig, "fig00_schematic"))
