"""Core definitions for the nonlocal (fractional-order) Ekman layer.

Non-equatorial f-plane, horizontally homogeneous, steady wind stress.
zeta = -z >= 0 is depth below the surface.  Complex velocity and complex
stress are psi = u + i v and T = T_x + i T_y, and the closure-free momentum
balance in flux form is

    i f rho psi = -dT/dzeta,        T(0) = tau,     T(inf) = 0.          (M)

Classical Ekman closes this with the local Fickian law T = -rho K dpsi/dzeta.
Here it is replaced by a Scott-Blair (fractional-order) flux-gradient law

    T = -rho K_gamma  D^gamma_zeta psi,        0 < gamma <= 1,           (C)

with D^gamma the Caputo derivative based at the surface.  gamma = 1 recovers
Ekman exactly.  K_gamma carries units m^(1+gamma) s^-1.

Angles are in degrees, measured from the wind-stress direction, negative to
the right (Northern Hemisphere, f > 0).
"""

import numpy as np
from scipy.special import gamma as gammafn

__all__ = ["Params", "characteristic_root", "deflection_angle", "depth_scale",
           "surface_velocity", "tail_amplitude", "total_turning",
           "turning_mod_2pi", "asymptotic_angle"]


class Params:
    """Model parameters.  Defaults are the non-dimensional reference case."""

    def __init__(self, gamma=0.7, f=1.0, K=1.0, rho=1.0, tau=1.0 + 0.0j):
        """Store the closure order and the physical constants."""
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must lie in (0, 1]")
        self.gamma = float(gamma)
        self.f = float(f)
        self.K = float(K)
        self.rho = float(rho)
        self.tau = complex(tau)

    @property
    def mu(self):
        """Order of the governing fractional equation, mu = 1 + gamma."""
        return 1.0 + self.gamma

    @property
    def b(self):
        """Coefficient of the characteristic equation p^mu = b."""
        return 1j * self.f / self.K

    def __repr__(self):
        """Return a reproducible representation of the parameters."""
        return (f"Params(gamma={self.gamma:g}, f={self.f:g}, K={self.K:g}, "
                f"rho={self.rho:g}, tau={self.tau:g})")


def characteristic_root(prm):
    """Return the principal (k = 0) root of p^mu = b.

    arg p0 = (pi/2)/mu lies in [pi/4, pi/2) for every gamma in (0, 1], so
    p0 is always in the right half plane and always carries a growing
    mode.  T(inf) = 0 is enforced by cancelling it in the numerator of
    the transform, which fixes psi(0).
    """
    return prm.b ** (1.0 / prm.mu)


def deflection_angle(gamma):
    r"""Surface deflection, -(pi/2) gamma/(1+gamma) in degrees.

    Equals -45 degrees at gamma = 1.
    """
    g = np.asarray(gamma, dtype=float)
    return -90.0 * g / (1.0 + g)


def depth_scale(prm):
    """Generalized Ekman depth (K_gamma/f)^(1/(1+gamma)).

    Reduces to sqrt(K/f) at gamma = 1.
    """
    return (prm.K / prm.f) ** (1.0 / prm.mu)


def surface_velocity(prm):
    """psi(0) = tau p0 / (i f rho), the value forced by the decay condition."""
    return prm.tau * characteristic_root(prm) / (1j * prm.f * prm.rho)


def tail_amplitude(prm):
    """Return the leading coefficient of the algebraic far field.

    The far field is T ~ A zeta^-gamma with

        A_gamma = tau (p0/b) / Gamma(1 - gamma).

    1/Gamma(1-gamma) -> 0 as gamma -> 1.  That is the mechanism by which
    the branch-cut tail disappears in the classical limit.
    """
    if prm.gamma >= 1.0:
        return 0.0 + 0.0j
    return (prm.tau * (characteristic_root(prm) / prm.b)
            / gammafn(1.0 - prm.gamma))


def asymptotic_angle(gamma):
    """Deep asymptotic direction, pi/(2 mu) - pi, in degrees."""
    g = np.asarray(gamma, dtype=float)
    return 90.0 / (1.0 + g) - 180.0


def turning_mod_2pi(gamma):
    """Return the exact modulo-2pi turning, -90 degrees.

    The unconditional statement is

        arg psi(inf) - arg psi(0) = -pi/2   (mod 2 pi)

    for every gamma in (0, 1).  In degrees, -90.

    This follows from the two closed-form directions alone and holds at every
    gamma.  It is a statement about the deep DIRECTION, not about how many
    times the velocity vector rotates on the way there.
    """
    g = np.atleast_1d(np.asarray(gamma, dtype=float))
    out = np.where(g >= 1.0, np.nan, -90.0)
    return out if out.size > 1 else float(out[0])


def total_turning(gamma, winding=0):
    """Return the net turning accumulated from surface to deep limit.

    The net turning is

        -90 - 360 n   degrees,

    where n = winding is the number of complete extra revolutions the velocity
    vector makes before settling onto the asymptotic direction.  n is a
    non-decreasing integer function of gamma alone, measured in
    scripts/fig03_angles.py and tabulated in outputs/reports/closed_forms.txt.
    It is 0 up to gamma of about 0.90, 1 from about 0.91 to 0.995, and at least
    2 by gamma = 0.999, and it diverges as gamma -> 1.

    Only the modulo-2pi statement is closed form.  The value of n is a measured
    property of the crossover between the exponentially decaying core and the
    algebraic tail, and no closed form for it is claimed here.
    """
    g = np.atleast_1d(np.asarray(gamma, dtype=float))
    out = np.where(g >= 1.0, -np.inf, -90.0 - 360.0 * np.asarray(winding))
    return out if out.size > 1 else float(out[0])
