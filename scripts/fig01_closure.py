"""Figure 1. The closure kernel, and the formulation it rules out.

(a) the memory kernel that weights the shear above the evaluation depth, made
dimensionless by the factor zeta^gamma and plotted against the dimensionless
separation (zeta - xi)/zeta.  The raw kernel carries units of m^-gamma, so it
cannot be compared across gamma without this normalization.
(b) a Caputo derivative based at the surface satisfies D^gamma psi (0) = 0 for
every psi with bounded derivative, so the stress computed from the closure
vanishes at the surface as zeta^(1-gamma) and T(0) = tau cannot be imposed on
psi.  The test profile is synthetic and its scale is arbitrary, so only the
slope carries meaning.  Fitted exponents are tabulated in
outputs/reports/figure_notes.txt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
from scipy.special import gamma as gammafn
from scipy.integrate import quad
import matplotlib.pyplot as plt

from bayu.plotting import (setup, GAMMA_COLORS, panel_label, gamma_handles,
                           outside_legend, save)
from bayu.io_utils import write_csv

GAMS = [0.2, 0.4, 0.6, 0.8, 0.95]
COLS = GAMMA_COLORS[:5]


def _dpsi_re(x):
    """Real part of the derivative of the synthetic test profile."""
    return -np.exp(-x) * (np.cos(x) + np.sin(x)) + 0.6 * x


def _dpsi_im(x):
    """Imaginary part of the derivative of the synthetic test profile."""
    return np.exp(-x) * (2.0 * np.cos(2.0 * x) - np.sin(2.0 * x))


def caputo_at(g, z):
    """Caputo derivative of the test profile at depth ``z``, order ``g``.

    The integrable endpoint singularity is removed by the substitution
    x = z (1 - s^(1/(1-g))), which maps the integrand to a smooth one.
    """
    a = 1.0 - g

    def component(deriv):
        def integrand(s):
            return deriv(z * (1.0 - s ** (1.0 / a))) * (z ** a / a)
        return quad(integrand, 0.0, 1.0, limit=200)[0]

    return (component(_dpsi_re)
            + 1j * component(_dpsi_im)) / gammafn(1.0 - g)


setup()
fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.0))

# ------------------------------------------------------------------ (a) kernel
ax = axes[0]
s = np.linspace(1e-3, 1.0, 600)
cols = {"s": s}
for g, col in zip(GAMS, COLS):
    k = s ** (-g) / gammafn(1.0 - g)
    ax.plot(s, k, color=col, lw=1.3)
    cols[f"kernel_g{g}"] = k
ax.set_xlim(0, 1)
ax.set_ylim(0.2, 40)
ax.set_yscale("log")
ax.set_xlabel(r"$(\zeta-\xi)/\zeta$")
ax.set_ylabel(r"$\zeta^{\gamma}(\zeta-\xi)^{-\gamma}/\Gamma(1-\gamma)$")
panel_label(ax, "(a)")

# -------------------------------------------------------------- (b) degeneracy
axb = axes[1]
zz = np.logspace(-5, -0.3, 22)
slopes, d2 = {}, {"zeta": zz}
for g, col in zip(GAMS, COLS):
    vals = np.array([abs(caputo_at(g, z)) for z in zz])
    axb.loglog(zz, vals, color=col, lw=1.2)
    slopes[g] = np.polyfit(np.log(zz[:12]), np.log(vals[:12]), 1)[0]
    d2[f"caputo_g{g}"] = vals
axb.set_xlabel(r"$\zeta$ (arbitrary units)")
axb.set_ylabel(r"$|\,{}^{C}\!D^{\gamma}_{\zeta}\psi\,|$ (arbitrary units)")
axb.set_xlim(zz[0], zz[-1])
panel_label(axb, "(b)")

fig.tight_layout()
handles, labels = gamma_handles(GAMS, COLS)
outside_legend(fig, handles, labels, ncol=5, y=0.0)
print(save(fig, "fig01_closure"))
write_csv("fig01a_kernel", cols)
write_csv("fig01b_caputo_degeneracy", d2)
write_csv("fig01b_slopes", {
    "gamma": np.array(GAMS),
    "slope_fitted": np.array([slopes[g] for g in GAMS]),
    "slope_predicted": 1.0 - np.array(GAMS),
})
for g in GAMS:
    print(f"  gamma={g:.2f}  fitted={slopes[g]:.6f}  predicted={1-g:.6f}  "
          f"|diff|={abs(slopes[g]-(1-g)):.2e}")
