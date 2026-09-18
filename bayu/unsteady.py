"""Spin-up: the time-dependent problem.

With the stress switched on at t = 0 and psi(zeta, 0) = 0,

    rho ( d psi/dt + i f psi ) = -d T/d zeta,    T(0, t) = tau H(t),

Laplace transforming in time replaces i f by sigma + i f everywhere, so the
entire depth structure of the steady problem carries over with

    b_sigma = (sigma + i f)/K_gamma,    p_sigma = b_sigma^(1/mu).

At the surface this collapses to one elementary transform,

    psitilde(0, sigma) = tau K^(-1/mu) (sigma + i f)^(1/mu - 1) / sigma,

whose inverse is exact:

    psi(0, t) = psi_steady * P(a, i f t),      a = gamma/(1 + gamma),

with P the regularized lower incomplete gamma function.  The exponent a is the
same combination of gamma that sets the deflection angle,
theta = -90 a degrees.

At gamma = 1 this is psi_steady erf(sqrt(i f t)), the classical impulsively
started Ekman surface current.  For gamma < 1 the residual inertial oscillation
decays algebraically with envelope exponent a - 1 = -1/(1 + gamma), so the
surface transient is not exponentially forgotten at any gamma.
"""

import numpy as np
from mpmath import mp, mpc, gammainc
from scipy.special import gamma as _sp_gamma
from .core import surface_velocity, characteristic_root
from .transform import talbot

__all__ = ["surface_spinup", "surface_spinup_talbot",
           "surface_spinup_quad", "profile_spinup",
           "transient_envelope_exponent"]


def _reg_lower_gamma(a, z):
    """Regularized lower incomplete gamma P(a, z) for complex z."""
    with mp.workdps(40):
        zz = mpc(complex(z).real, complex(z).imag)
        return complex(gammainc(a, 0, zz, regularized=True))


def surface_spinup(prm, t):
    """psi(0, t) in closed form."""
    a = prm.gamma / prm.mu
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ps = surface_velocity(prm)
    out = np.empty(t.shape, dtype=complex)
    for j, tj in enumerate(t):
        out[j] = 0.0 if tj <= 0 else ps * _reg_lower_gamma(a, 1j * prm.f * tj)
    return out


def surface_spinup_talbot(prm, t, M=32):
    """Return psi(0, t) by numerical inversion of psitilde(0, sigma).

    This is an independent check on the closed form.

    The Talbot contour has half-width r pi = 2 pi M/(5 f t) in the imaginary
    direction, so it encloses the branch point at sigma = -i f only while
    f t < 2 pi M/5, about 40 at M = 32.  Beyond that the inversion silently
    drops the branch point and is not a valid check; use surface_spinup_quad.
    """
    a = prm.gamma / prm.mu

    def F(s):
        return (prm.tau * prm.K ** (-1.0 / prm.mu)
                * (s + 1j * prm.f) ** (1.0 / prm.mu - 1.0) / (s * prm.rho))
    return talbot(F, t, M=M)


def surface_spinup_quad(prm, t):
    """Return psi(0, t) by direct quadrature of the convolution.

    The convolution is

        psi(0,t) = psi_steady (i f)^a / Gamma(a) int_0^t e^{-i f s} s^{a-1} ds,

    which is independent of the incomplete-gamma implementation and valid at
    any t.
    """
    from scipy.integrate import quad as _quad
    a = prm.gamma / prm.mu
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ps = surface_velocity(prm)
    pref = (1j * prm.f) ** a / _sp_gamma(a)
    out = np.empty(t.shape, dtype=complex)
    for j, tj in enumerate(t):
        # s = tj * w^(1/a) removes the endpoint singularity at s = 0
        fr = _quad(lambda w: np.cos(prm.f * tj * w ** (1.0 / a)) * tj ** a / a,
                   0, 1, limit=400)[0]
        fi = _quad(
            lambda w: -np.sin(prm.f * tj * w ** (1.0 / a)) * tj ** a / a,
            0, 1, limit=400)[0]
        out[j] = ps * pref * (fr + 1j * fi)
    return out


def profile_spinup(prm, zeta, t, M=24):
    """Return psi(zeta, t) by nested contour inversion.

    The depth transform is inverted on the inner contour and the time
    transform on the outer one.
    """
    zeta = np.atleast_1d(np.asarray(zeta, dtype=float))

    def psi_tilde(sig_scalar):
        bs = (sig_scalar + 1j * prm.f) / prm.K
        ps = bs ** (1.0 / prm.mu)
        tl = prm.tau / sig_scalar

        def That(p):
            return tl * p ** (prm.gamma - 1.0) * (p - ps) / (p ** prm.mu - bs)

        def Phat(p):
            return -(p * That(p) - tl) / (prm.rho * (sig_scalar + 1j * prm.f))

        out = np.empty(zeta.shape, dtype=complex)
        pos = zeta > 0
        out[~pos] = tl * ps / (prm.rho * (sig_scalar + 1j * prm.f))
        if np.any(pos):
            out[pos] = talbot(Phat, zeta[pos], M=M)
        return out

    k = np.arange(1, M)
    th = k * np.pi / M
    cot = 1.0 / np.tan(th)
    w = 1.0 + 1j * (th + (th * cot - 1.0) * cot)
    r = 2.0 * M / (5.0 * t)
    s = r * th * (cot + 1j)
    acc = 0.5 * np.exp(r * t) * psi_tilde(r + 0j)
    for kk in range(M - 1):
        acc += 0.5 * (np.exp(s[kk] * t) * psi_tilde(s[kk]) * w[kk]
                      + np.exp(np.conj(s[kk]) * t) * psi_tilde(np.conj(s[kk]))
                      * np.conj(w[kk]))
    return (r / M) * acc


def transient_envelope_exponent(gamma):
    """Algebraic decay exponent of the surface transient, a - 1.

    Equals -1/(1 + gamma).
    """
    g = np.asarray(gamma, dtype=float)
    return -1.0 / (1.0 + g)
