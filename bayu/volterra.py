r"""Independent numerical solution by product integration.

Applying the fractional integral I^gamma to the closure (C) and integrating the
momentum balance (M) turns the boundary-value problem into a coupled weakly
singular Volterra system with no derivatives left in it:

    psi(zeta) = psi0 - (rho K_gamma)^-1 I^gamma T(zeta),
    T(zeta)   = tau - i f rho \\int_0^zeta psi(xi) dxi,

with psi0 = tau p0/(i f rho) supplied by the far-field condition.  The
fractional integral is discretized with the product trapezoidal weights of
Diethelm, the plain integral with the trapezoidal rule, and the resulting
2x2 linear system at each step is solved directly rather than iterated.

The scheme is an independent third algorithm: it never evaluates a Mittag-
Leffler function and never touches the Laplace plane.  Its observed order is
min(2, 1 + gamma), so the measured order is itself a test of the formulation.
"""

import numpy as np
from scipy.special import gamma as gammafn
from .core import surface_velocity

__all__ = ["pi_weights", "solve_volterra"]


def pi_weights(n, g):
    """Product trapezoidal weights a_{j,n} for I^gamma on a uniform grid."""
    j = np.arange(n + 1, dtype=float)
    m = n - j
    a = np.empty(n + 1)
    if n == 0:
        return np.array([0.0])
    a[0] = (n - 1.0) ** (g + 1.0) - n ** g * (n - g - 1.0)
    if n > 1:
        mi = m[1:n]
        a[1:n] = (mi + 1.0) ** (g + 1.0) + (mi - 1.0) ** (g + 1.0) \
            - 2.0 * mi ** (g + 1.0)
    a[n] = 1.0
    return a


def solve_volterra(prm, zmax, N, psi0=None):
    """
    March the Volterra system on a uniform grid of N intervals over [0, zmax].

    Returns (zeta, psi, T).
    """
    g, h = prm.gamma, zmax / N
    w = h ** g / gammafn(g + 2.0)
    c = 1.0 / (prm.rho * prm.K)
    if psi0 is None:
        psi0 = surface_velocity(prm)

    z = np.linspace(0.0, zmax, N + 1)
    psi = np.zeros(N + 1, dtype=complex)
    T = np.zeros(N + 1, dtype=complex)
    psi[0], T[0] = psi0, prm.tau
    Q = 0.0 + 0.0j                                    # running integral of psi

    lhs = 1.0 - 1j * prm.f * w * h * c * prm.rho / 2.0
    for n in range(1, N + 1):
        a = pi_weights(n, g)
        S = w * np.dot(a[:n], T[:n])                  # history of I^gamma T
        Tpred = prm.tau - 1j * prm.f * prm.rho * (Q + 0.5 * h * psi[n - 1])
        rhs = psi0 - c * (S + w * Tpred)
        psi[n] = rhs / lhs
        T[n] = (prm.tau - 1j * prm.f * prm.rho
                * (Q + 0.5 * h * (psi[n - 1] + psi[n])))
        Q += 0.5 * h * (psi[n - 1] + psi[n])
    return z, psi, T
