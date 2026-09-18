"""Closed-form solution, evaluated by contour inversion.

Laplace transforming (M) and (C) in depth, with b = i f / K_gamma and
mu = 1 + gamma,

    That(p) = [ tau p^gamma - i f rho psi0 p^(gamma-1) ] / ( p^mu - b ).

The far-field condition T(inf) = 0 forces the numerator to vanish at the
right-half-plane root p0 = b^(1/mu), which gives psi0 = tau p0 / (i f rho) and

    That(p) = tau p^(gamma-1) (p - p0) / (p^mu - b),
    psihat(p) = -[ p That(p) - tau ] / (i f rho).
"""

import numpy as np
from .core import characteristic_root
from .transform import talbot

# Below this depth the Talbot contour radius r = 2M/(5 zeta) becomes large
# enough that p^gamma overflows the dynamic range and the inversion loses all
# accuracy.  The Mittag-Leffler series is cheap and exact there, since
# |b zeta^mu| is tiny, so the two are spliced.  The splice point is inside
# the region where both are accurate to 1e-12; see fig06_verification.
ZCROSS = 5.0e-2

__all__ = ["stress_hat", "velocity_hat", "stress", "velocity",
           "stress_uncancelled", "velocity_asymptote", "velocity_contour"]


def stress_hat(prm, p):
    """That(p), with the growing mode cancelled."""
    p0 = characteristic_root(prm)
    return prm.tau * p ** (prm.gamma - 1.0) * (p - p0) / (p ** prm.mu - prm.b)


def velocity_hat(prm, p):
    """psihat(p) = -(p That - tau)/(i f rho)."""
    return -(p * stress_hat(prm, p) - prm.tau) / (1j * prm.f * prm.rho)


def stress(prm, zeta, M=32):
    """T(zeta), spliced: series for zeta < ZCROSS, contour inversion above."""
    from .mittagleffler import stress_series
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    out = np.empty(zeta.shape, dtype=complex)
    lo = zeta < ZCROSS
    if np.any(lo):
        out[lo] = stress_series(prm, zeta[lo])
    if np.any(~lo):
        out[~lo] = talbot(lambda p: stress_hat(prm, p), zeta[~lo], M=M)
    return out


def velocity(prm, zeta, M=32):
    """Velocity: series below ZCROSS, contour inversion above."""
    from .mittagleffler import velocity_series
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    out = np.empty(zeta.shape, dtype=complex)
    lo = zeta < ZCROSS
    if np.any(lo):
        out[lo] = velocity_series(prm, zeta[lo])
    if np.any(~lo):
        out[~lo] = talbot(lambda p: velocity_hat(prm, p), zeta[~lo], M=M)
    return out


def velocity_contour(prm, zeta, M=32):
    """Contour inversion alone, with no splice, for cross-method comparison."""
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    return talbot(lambda p: velocity_hat(prm, p), zeta, M=M)


def stress_uncancelled(prm, zeta, M=32):
    """Return the branch obtained without cancelling p0.

    Retained only to show what the far-field condition excludes.  It
    diverges with depth.
    """
    def Fh(p):
        return prm.tau * p ** (prm.gamma) / (p ** prm.mu - prm.b)
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    out = np.empty(zeta.shape, dtype=complex)
    pos = zeta > 0
    out[~pos] = prm.tau
    if np.any(pos):
        out[pos] = talbot(Fh, zeta[pos], M=M)
    return out


def velocity_asymptote(prm, zeta):
    """Return the leading algebraic far field of the velocity.

    The far field is

        psi ~ gamma A_gamma zeta^(-1-gamma) / (i f rho),
        A_gamma = tau (p0/b)/Gamma(1-gamma),

    whose argument is arg tau + pi/(2 mu) - pi, independent of depth.
    """
    from .core import tail_amplitude
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    A = tail_amplitude(prm)
    return prm.gamma * A * zeta ** (-1.0 - prm.gamma) / (1j * prm.f * prm.rho)
