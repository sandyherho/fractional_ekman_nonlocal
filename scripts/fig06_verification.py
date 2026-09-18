"""Figure 6. Numerical verification.

Three independent algorithms are used and no two share a code path: contour
inversion of the depth transform, arbitrary-precision Mittag-Leffler summation,
and product-integration marching of the equivalent Volterra system.

(a) self-convergence of the contour rule in the node count.
(b) agreement between contour inversion and the series across gamma.
(c) convergence of the product-integration march against the contour solution,
with reference slopes min(2, 1 + gamma) (dotted).
(d) residual of the Ekman transport invariant, relative to |tau|/(rho f) and
therefore dimensionless.  The invariant follows from the momentum balance for
any closure, so its residual is a property of the solution method alone.
Measured values are tabulated in outputs/reports/verification.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bayu.core import Params
from bayu.solution import velocity, stress, velocity_contour
from bayu.mittagleffler import velocity_series
from bayu.volterra import solve_volterra
from bayu.plotting import (setup, OKABE_ITO as C, GAMMA_COLORS, panel_label,
                           gamma_handles, outside_legend, save)
from bayu.io_utils import write_csv

setup()
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4))
GAMS = [0.3, 0.5, 0.7, 0.9]
COLS = GAMMA_COLORS[:4]

# --------------------------------------------------- (a) contour self-converge
ax = axes[0, 0]
zt = np.array([0.5, 2.0, 8.0])
Ms = np.arange(8, 65, 4)
prm = Params(gamma=0.6)
ref = velocity_series(prm, zt)
errM = np.array([np.max(np.abs(velocity_contour(prm, zt, M=int(m)) - ref))
                 for m in Ms])
ax.semilogy(Ms, errM, "o-", ms=3.5, color=C["blue"], lw=1.1)
ax.axvline(32, color=C["grey"], lw=0.7, ls=":")
ax.set_xlabel(r"contour nodes $M$")
ax.set_ylabel(r"$\max_\zeta|\psi_{\mathrm{ct}}-\psi_{\mathrm{ML}}|$")
ax.set_ylim(1e-13, 1e-1)
panel_label(ax, "(a)")

# ------------------------------------------------- (b) cross-method agreement
axb = axes[0, 1]
gg = np.linspace(0.1, 1.0, 19)
zc = np.array([0.25, 1.0, 4.0, 16.0])
cross = np.array([np.max(np.abs(velocity_contour(Params(gamma=g), zc)
                                - velocity_series(Params(gamma=g), zc)))
                  for g in gg])
axb.semilogy(gg, cross, "o-", ms=3.5, color=C["green"], lw=1.1)
axb.set_xlabel(r"$\gamma$")
axb.set_ylabel(r"$\max_\zeta|\psi_{\mathrm{ct}}-\psi_{\mathrm{ML}}|$")
axb.set_ylim(1e-11, 1e-8)
panel_label(axb, "(b)")

# ------------------------------------------------------ (c) Volterra order
axc = axes[1, 0]
Ns = np.array([100, 200, 400, 800, 1600, 3200])
h = 4.0 / Ns
orders, conv = {}, {"h": h}
for g, col in zip(GAMS, COLS):
    prm = Params(gamma=g)
    e = []
    for N in Ns:
        z, ps, _ = solve_volterra(prm, 4.0, int(N))
        e.append(np.max(np.abs(ps[1:] - velocity(prm, z[1:]))))
    e = np.array(e)
    conv[f"err_g{g}"] = e
    orders[g] = np.polyfit(np.log(h[-4:]), np.log(e[-4:]), 1)[0]
    axc.loglog(h, e, "o-", ms=3.2, color=col, lw=1.1)
    axc.loglog(h, 0.30 * e[0] * (h / h[0]) ** (1.0 + g), color=col, lw=0.6,
               ls=":")
axc.set_xlabel(r"$h/\delta_\gamma$")
axc.set_ylabel(r"$\max_\zeta|\psi_{\mathrm{PI}}-\psi_{\mathrm{ct}}|$")
panel_label(axc, "(c)")

# ---------------------------------------------------- (d) transport invariant
axd = axes[1, 1]
gt = np.linspace(0.1, 1.0, 19)
Z = 40.0
uq = np.linspace(0.0, Z ** (1.0 / 3.0), 6001)
zq = uq ** 3
tr_pi, tr_ct = [], []
for g in gt:
    prm = Params(gamma=g)
    Iq = np.trapezoid(3.0 * uq ** 2 * velocity(prm, zq), uq) \
        + stress(prm, Z)[0] / (1j * prm.f * prm.rho)
    ref = abs(prm.tau) / (prm.rho * prm.f)
    tr_ct.append(abs(Iq + 1j * prm.tau / (prm.rho * prm.f)) / ref)
    z, pv, T = solve_volterra(prm, Z, 8000)
    Ip = np.trapezoid(pv, z) + T[-1] / (1j * prm.f * prm.rho)
    tr_pi.append(abs(Ip + 1j * prm.tau / (prm.rho * prm.f)) / ref)
tr_ct, tr_pi = np.array(tr_ct), np.array(tr_pi)
axd.semilogy(gt, tr_ct, "o-", ms=3.2, color=C["blue"], lw=1.1)
axd.semilogy(gt, tr_pi, "s-", ms=3.2, color=C["vermil"], lw=1.1)
axd.set_xlabel(r"$\gamma$")
axd.set_ylabel(r"$\rho f|\tau|^{-1}"
               r"|\int_0^\infty\psi\,d\zeta + i\tau/(\rho f)|$")
axd.set_ylim(1e-17, 1e-8)
panel_label(axd, "(d)")

fig.tight_layout(h_pad=2.0)
hg, lg = gamma_handles(GAMS, COLS)
hg += [Line2D([], [], color=C["grey"], lw=0.6, ls=":"),
       Line2D([], [], color=C["blue"], marker="o", ms=3.2, lw=1.1),
       Line2D([], [], color=C["vermil"], marker="s", ms=3.2, lw=1.1)]
lg += [r"$\min(2,1+\gamma)$", "contour", "product integration"]
outside_legend(fig, hg, lg, ncol=4, y=0.0)
print(save(fig, "fig06_verification"))
write_csv("fig06a_contour_nodes", {"M": Ms.astype(float), "err": errM})
write_csv("fig06b_cross_method", {"gamma": gg, "err": cross})
write_csv("fig06c_volterra_order", conv)
write_csv("fig06d_transport", {"gamma": gt, "transport_err_contour": tr_ct,
                               "transport_err_pi": tr_pi})
for g in GAMS:
    print(f"  gamma={g:.1f}  observed={orders[g]:.4f}  "
          f"predicted={min(2.0, 1 + g):.4f}")
