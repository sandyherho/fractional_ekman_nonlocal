"""Numerical Laplace inversion, complex-valued.

The fixed-Talbot rule of Abate and Valko deforms the Bromwich line onto the
cotangent contour

    s(theta) = r theta (cot theta + i),    theta in (-pi, pi),   r = 2M/(5t),

which wraps the negative real axis.  It therefore handles the branch point of
p^mu at the origin and the cut along the negative axis with no special
treatment.  That is what is needed here: once the growing mode is cancelled the
transform has no poles at all for gamma < 1/2, and the whole solution is the
branch-cut contribution.

Standard implementations halve the work by taking a real part, which assumes
f(t) real and hence F(conj s) = conj F(s).  The stress and velocity here are
complex-valued functions of a real depth and carry no such symmetry, so the
full contour is summed over theta of both signs and no symmetry is imposed.
"""

import numpy as np

__all__ = ["talbot"]


def talbot(F, t, M=32):
    """
    Invert the Laplace transform F at the points t.

    Parameters
    ----------
    F : callable
        F(p) for a 1-D complex array p.  May be complex-valued with no
        conjugate symmetry.
    t : array_like of positive floats
    M : int
        Contour nodes per half.  Convergence is geometric in M; M = 32 is at
        the roundoff-limited plateau for the transforms used here.

    Returns
    -------
    complex ndarray, shape of t.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t <= 0):
        raise ValueError("talbot requires t > 0")

    k = np.arange(1, M)
    th = k * np.pi / M
    cot = 1.0 / np.tan(th)
    sig = th + (th * cot - 1.0) * cot          # sigma(theta)
    w = 1.0 + 1j * sig                          # contour weight

    out = np.empty(t.shape, dtype=complex)
    for j, tj in enumerate(t):
        r = 2.0 * M / (5.0 * tj)
        s = r * th * (cot + 1j)                 # upper half of the contour
        sc = np.conj(s)                         # lower half
        acc = 0.5 * np.exp(r * tj) * F(np.array([r + 0j]))[0]
        acc += 0.5 * np.sum(np.exp(s * tj) * F(s) * w
                            + np.exp(sc * tj) * F(sc) * np.conj(w))
        out[j] = (r / M) * acc
    return out
