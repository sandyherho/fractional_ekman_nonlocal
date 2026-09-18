"""Figure 3. The angular structure.

(a) surface deflection angle and deep asymptotic direction against gamma, with
the band between them.  Markers are measured from the arbitrary-precision
series, lines are closed form.
(b) cumulative turning of psi with depth.  The deep direction is a quarter turn
from the surface direction modulo 2 pi at every gamma, but the number of
complete revolutions made on the way there is a non-decreasing integer n(gamma)
that diverges as gamma -> 1: gamma = 0.95 completes one extra turn.
Values of n and the overshoot are tabulated in
outputs/reports/figure_notes.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bayu.core import Params, deflection_angle, asymptotic_angle
from bayu.solution import velocity
from bayu.mittagleffler import velocity_series
from bayu.plotting import (setup, OKABE_ITO as C, GAMMA_COLORS, panel_label,
                           gamma_handles, outside_legend, save)
from bayu.io_utils import write_csv


def turning(prm, z):
    """Cumulative turning of psi in degrees, unwrapped from the surface."""
    ps = velocity(prm, z)
    return np.degrees(np.unwrap(np.angle(ps)) - np.angle(ps[0]))


setup()
fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.2))

# ------------------------------------------------------------------- (a) angle
ax = axes[0]
gg = np.linspace(0.02, 1.0, 400)
ax.fill_between(gg, deflection_angle(gg), asymptotic_angle(gg),
                color=C["skyblue"], alpha=0.20, lw=0)
ax.plot(gg, deflection_angle(gg), color=C["blue"], lw=1.4)
ax.plot(gg, asymptotic_angle(gg), color=C["blue"], lw=1.4, ls="--")
gm = np.arange(0.1, 1.01, 0.1)
meas = np.array([
    np.degrees(np.angle(velocity_series(Params(gamma=g), 1e-200)[0]))
    for g in gm])
ax.plot(gm, meas, "o", ms=3.8, mfc="none", mec=C["vermil"], mew=1.0, ls="none")
ax.axhline(-45.0, color=C["grey"], lw=0.6, ls=":")
ax.set_xlim(0, 1)
ax.set_ylim(-185, 5)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel("angle from wind stress (deg)")
ax.set_yticks([0, -45, -90, -135, -180])
panel_label(ax, "(a)")

ha = [Line2D([], [], color=C["blue"], lw=1.4),
      Line2D([], [], color=C["blue"], lw=1.4, ls="--"),
      Line2D([], [], color=C["vermil"], marker="o", ms=3.8, mfc="none",
             mew=1.0, ls="none"),
      Line2D([], [], color=C["grey"], lw=0.6, ls=":")]
la = [r"$\arg\psi(0)$", r"$\arg\psi(\infty)$", "series", r"$-45^\circ$"]

# ----------------------------------------------------------------- (b) turning
axb = axes[1]
z = np.logspace(-4, 4.0, 1400)
GAMS = [0.2, 0.5, 0.7, 0.9, 0.95, 1.0]
tab = {"zeta": z}
for g, col in zip(GAMS, GAMMA_COLORS):
    t = turning(Params(gamma=g), z)
    tab[f"turn_g{g}"] = t
    axb.semilogx(z, t, color=col, lw=1.3)
axb.axhline(-90.0, color=C["grey"], lw=0.7, ls=":")
axb.set_xlim(z[0], z[-1])
axb.set_ylim(-500, 20)
axb.set_xlabel(r"$\zeta/\delta_\gamma$")
axb.set_ylabel("cumulative turning (deg)")
axb.set_yticks([0, -90, -180, -270, -360, -450])
panel_label(axb, "(b)")

fig.tight_layout(h_pad=1.6)
hb, lb = gamma_handles(GAMS, GAMMA_COLORS)
leg1 = fig.legend(ha, la, loc="upper center", bbox_to_anchor=(0.5, 0.005),
                  ncol=4, frameon=False, handlelength=1.8, columnspacing=1.4,
                  handletextpad=0.5, borderaxespad=0.0)
fig.add_artist(leg1)
fig.legend(hb, lb, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=6,
           frameon=False, handlelength=1.6, columnspacing=1.0,
           handletextpad=0.4, borderaxespad=0.0)
print(save(fig, "fig03_angles"))
write_csv("fig03a_angles", {"gamma": gg, "theta": deflection_angle(gg),
                            "theta_inf": asymptotic_angle(gg)})
write_csv("fig03a_measured", {"gamma": gm, "theta_measured": meas,
                              "theta_closed": deflection_angle(gm)})
write_csv("fig03b_turning", tab)
print("max |measured - closed form|:",
      f"{np.max(np.abs(meas - deflection_angle(gm))):.3e} deg")
gw = np.array([0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.91, 0.95, 0.99, 0.995, 0.999])
wind, nets = [], []
for g in gw:
    t = turning(Params(gamma=g), z)
    n = int(round((-t[-1] - 90.0) / 360.0))
    wind.append(n)
    nets.append(t[-1])
    print(f"  gamma={g:.3f}  net turning {t[-1]:+10.3f} deg   winding n={n}   "
          f"residual from -90-360n: {t[-1] + 90.0 + 360.0 * n:+.4f} deg   "
          f"overshoot {max(0.0, -t.min() - 90.0):.3f} deg")
write_csv("fig03c_winding", {"gamma": gw, "net_turning": np.array(nets),
                             "winding_number": np.array(wind, dtype=float)})
