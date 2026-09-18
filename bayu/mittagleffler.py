"""Two-parameter Mittag-Leffler function E_{mu,nu}(z), arbitrary precision.

Used only as an independent reference for the Talbot inversion at moderate
argument.  The defining series converges for every z but loses relative
precision through subtractive cancellation once |z| is large, so it is summed
in mpmath at a working precision chosen from |z| rather than in double
precision.  The cross-check in scripts/fig06_verification.py is therefore a
comparison of two genuinely independent algorithms, not a self-test.
"""

import numpy as np
from mpmath import mp, mpc, mpf, gamma as mpgamma

__all__ = ["mittag_leffler", "stress_series", "velocity_series"]


def mittag_leffler(mu, nu, z, dps=None, nmax=None):
    """E_{mu,nu}(z) by direct summation at adaptive working precision."""
    az = abs(complex(z))
    if dps is None:
        dps = int(40 + 2.2 * az)
    if nmax is None:
        nmax = int(60 + 3.0 * az)
    with mp.workdps(dps):
        zz = mpc(complex(z).real, complex(z).imag)
        mmu, nnu = mpf(mu), mpf(nu)
        s, term = mpc(0), mpc(1)
        for k in range(nmax):
            s += term / mpgamma(mmu * k + nnu)
            term *= zz
        return complex(s)


def stress_series(prm, zeta):
    """Return the stress from the term-by-term series inversion.

    T(zeta) = tau [ E_{mu,1}(b z^mu) - p0 z E_{mu,2}(b z^mu) ],

    the term-by-term inversion of That(p) using
    L^-1{ p^(mu-nu)/(p^mu - b) } = zeta^(nu-1) E_{mu,nu}(b zeta^mu).
    """
    from .core import characteristic_root
    mu, b, p0 = prm.mu, prm.b, characteristic_root(prm)
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    out = np.empty(zeta.shape, dtype=complex)
    for j, z in enumerate(zeta):
        if z == 0.0:
            out[j] = prm.tau
            continue
        arg = b * z ** mu
        out[j] = prm.tau * (mittag_leffler(mu, 1.0, arg)
                            - p0 * z * mittag_leffler(mu, 2.0, arg))
    return out


def velocity_series(prm, zeta):
    """Return the velocity from the differentiated series.

    psi(zeta) = -T'(zeta)/(i f rho) with

        T'(zeta) = tau [ b zeta^(mu-1) E_{mu,mu}(b zeta^mu)
                         - p0 E_{mu,1}(b zeta^mu) ],

    obtained by differentiating the series termwise.  At zeta = 0 the first
    term carries zeta^gamma and vanishes, leaving psi(0) = tau p0/(i f rho).
    """
    from .core import characteristic_root
    mu, b, p0 = prm.mu, prm.b, characteristic_root(prm)
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
    out = np.empty(zeta.shape, dtype=complex)
    for j, z in enumerate(zeta):
        if z == 0.0:
            out[j] = prm.tau * p0 / (1j * prm.f * prm.rho)
            continue
        arg = b * z ** mu
        Tp = prm.tau * (b * z ** (mu - 1.0) * mittag_leffler(mu, mu, arg)
                        - p0 * mittag_leffler(mu, 1.0, arg))
        out[j] = -Tp / (1j * prm.f * prm.rho)
    return out
