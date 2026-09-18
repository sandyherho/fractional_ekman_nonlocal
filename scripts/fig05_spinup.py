"""Figure 5. Spin-up of the surface current.

(a) the surface trajectory in the velocity plane, normalized by its own steady
value.
(b) the residual, with the closed-form algebraic envelope (dotted).  Exponents
and residuals are tabulated in outputs/reports/figure_notes.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
from scipy.special import gamma as gammafn, erf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bayu.core import Params, surface_velocity
from bayu.unsteady import (surface_spinup, surface_spinup_talbot,
                           surface_spinup_quad, transient_envelope_exponent)
from bayu.plotting import (setup, OKABE_ITO as C, GAMMA_COLORS, panel_label,
                           gamma_handles, outside_legend, save)
from bayu.io_utils import write_csv

GAMS = [0.2, 0.5, 0.8, 1.0]
COLS = [GAMMA_COLORS[0], GAMMA_COLORS[1], GAMMA_COLORS[3], C["black"]]

setup()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
t = np.logspace(-3, 2.6, 1400)
tab = {"ft": t}

for g, col in zip(GAMS, COLS):
    prm = Params(gamma=g)
    r = surface_spinup(prm, t) / surface_velocity(prm)
    tab[f"spinup_g{g}"] = r
    axes[0].plot(r.real, r.imag, color=col, lw=1.2)
    axes[1].loglog(t, np.abs(1.0 - r), color=col, lw=1.3)
    a = g / (1.0 + g)
    ta = t[t > 0.7]
    axes[1].loglog(ta, ta ** (a - 1.0) / gammafn(a), color=col, lw=0.7, ls=":")

axes[0].plot([1.0], [0.0], "o", ms=4, color=C["vermil"], zorder=5)
axes[0].axhline(0, color=C["grey"], lw=0.5, zorder=0)
axes[0].axvline(0, color=C["grey"], lw=0.5, zorder=0)
axes[0].set_aspect("equal")
axes[0].set_xlim(-0.35, 1.45)
axes[0].set_ylim(-0.55, 0.95)
axes[0].set_xlabel(r"$\mathrm{Re}\,[\psi(0,t)/\psi(0,\infty)]$")
axes[0].set_ylabel(r"$\mathrm{Im}\,[\psi(0,t)/\psi(0,\infty)]$")
panel_label(axes[0], "(a)")

axes[1].set_xlim(t[0], t[-1])
axes[1].set_ylim(2e-3, 2e1)
axes[1].set_xlabel(r"$f t$")
axes[1].set_ylabel(r"$|1-\psi(0,t)/\psi(0,\infty)|$")
panel_label(axes[1], "(b)")

fig.tight_layout()
handles, labels = gamma_handles(GAMS, COLS)
handles += [
    Line2D([], [], color=C["grey"], lw=0.7, ls=":"),
    Line2D([], [], color=C["vermil"], marker="o", ms=4, ls="none"),
]
labels += [r"$(ft)^{a-1}/\Gamma(a)$", "steady"]
outside_legend(fig, handles, labels, ncol=6, y=0.0)
print(save(fig, "fig05_spinup"))
write_csv("fig05_spinup", tab)

tc = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
tq = np.array([0.1, 1.0, 10.0, 100.0, 400.0])
for g in GAMS:
    prm = Params(gamma=g)
    d1 = np.max(np.abs(surface_spinup(prm, tc)
                       - surface_spinup_talbot(prm, tc)))
    d2 = np.max(np.abs(surface_spinup(prm, tq) - surface_spinup_quad(prm, tq))
                / np.abs(surface_spinup(prm, tq)))
    print(f"  gamma={g:.1f}  contour {d1:.3e}   quadrature (rel) {d2:.3e}")
prm = Params(gamma=1.0)
classical = np.max(np.abs(surface_spinup(prm, tq)
                          - surface_velocity(prm)
                          * erf(np.sqrt(1j * tq))))
print(f"  classical erf check: {classical:.3e}")
