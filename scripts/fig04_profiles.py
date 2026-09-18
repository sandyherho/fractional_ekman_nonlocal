"""Figure 4. Decay of the velocity with depth.

(a) speed profiles with the closed-form algebraic tails (dotted), which carry
no fitted constant.
(b) the same profiles compensated by zeta^(1+gamma), each flattening onto its
predicted plateau (dotted).  Measured amplitudes are tabulated in
outputs/reports/figure_notes.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bayu.core import Params, tail_amplitude
from bayu.solution import velocity, velocity_asymptote
from bayu.plotting import (setup, OKABE_ITO as C, GAMMA_COLORS, panel_label,
                           gamma_handles, outside_legend, save)
from bayu.io_utils import write_csv

GAMS = [0.2, 0.4, 0.6, 0.8, 1.0]
COLS = GAMMA_COLORS[:4] + [C["black"]]

setup()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
z = np.logspace(-1.2, 2.6, 520)
tab = {"zeta": z}

for g, col in zip(GAMS, COLS):
    prm = Params(gamma=g)
    p0 = abs(velocity(prm, 1e-12)[0])
    sp = np.abs(velocity(prm, z)) / p0
    tab[f"speed_g{g}"] = sp
    axes[0].loglog(z, sp, color=col, lw=1.3)
    if g < 1.0:
        axes[0].loglog(z, np.abs(velocity_asymptote(prm, z)) / p0, color=col,
                       lw=0.7, ls=":")
        axes[1].semilogx(z, sp * z ** (1.0 + g), color=col, lw=1.3)
        axes[1].axhline(np.abs(tail_amplitude(prm)) * g / p0, color=col,
                        lw=0.7, ls=":")
        tab[f"comp_g{g}"] = sp * z ** (1.0 + g)
    else:
        axes[1].semilogx(z, sp * z ** 2.0, color=col, lw=1.3)
        tab["comp_g1.0"] = sp * z ** 2.0

axes[0].set_xlim(z[0], z[-1])
axes[0].set_ylim(1e-8, 3)
axes[0].set_xlabel(r"$\zeta/\delta_\gamma$")
axes[0].set_ylabel(r"$|\psi|/|\psi_0|$")
panel_label(axes[0], "(a)")

axes[1].set_xlim(z[0], z[-1])
axes[1].set_yscale("log")
axes[1].set_ylim(1e-3, 5)
axes[1].set_xlabel(r"$\zeta/\delta_\gamma$")
axes[1].set_ylabel(r"$(\zeta/\delta_\gamma)^{1+\gamma}\,|\psi|/|\psi_0|$")
panel_label(axes[1], "(b)")

fig.tight_layout()
handles, labels = gamma_handles(GAMS, COLS)
handles.append(Line2D([], [], color=C["grey"], lw=0.7, ls=":"))
labels.append("closed-form tail")
outside_legend(fig, handles, labels, ncol=6, y=0.0)
print(save(fig, "fig04_profiles"))
write_csv("fig04_profiles", tab)

zd = np.array([1e2, 1e3, 1e4, 1e5])
rows = []
for g in GAMS[:-1]:
    prm = Params(gamma=g)
    v = np.abs(velocity(prm, zd)) * zd ** (1.0 + g)
    pred = g * np.abs(tail_amplitude(prm)) / prm.f
    rows.append((g, v, pred, abs(v[-1] / pred - 1.0)))
    print(f"   {g:.1f}   | " + "  ".join(f"{x:.6f}" for x in v)
          + f" |  {pred:.6f}  | {rows[-1][3]:.2e}")
write_csv("fig04_tail_amplitude",
          {"gamma": np.array([r[0] for r in rows]),
           "amp_1e2": np.array([r[1][0] for r in rows]),
           "amp_1e3": np.array([r[1][1] for r in rows]),
           "amp_1e4": np.array([r[1][2] for r in rows]),
           "amp_1e5": np.array([r[1][3] for r in rows]),
           "amp_closed": np.array([r[2] for r in rows]),
           "rel_err": np.array([r[3] for r in rows])})
