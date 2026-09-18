"""Static prose for the plain-text reports.

Keeping the narrative text here rather than inline in the report generator
separates what is written from what is computed, and keeps both files within
the PEP 8 line limit.  Every block is stored unindented and is indented by two
spaces when rendered.
"""

from textwrap import dedent

__all__ = ["PROSE", "block"]


def block(key):
    """Return a prose block as a list of report lines."""
    text = dedent(PROSE[key]).strip("\n")
    return ["  " + line if line else "" for line in text.split("\n")]


PROSE = {

    "winding_note": """
        The residual is set by truncation at finite depth, not by the
        identity: it shrinks monotonically as the deep limit is
        approached and is below 0.01 degrees wherever the algebraic tail
        has taken over.

        n is measured, not derived.  It counts the complete revolutions
        the velocity vector makes before the algebraic tail fixes its
        direction, and is governed by the crossover between the
        exponentially decaying core and that tail.  Since the tail
        amplitude carries 1/Gamma(1-gamma) and vanishes at gamma = 1, the
        crossover depth and therefore n grow without bound as
        gamma -> 1.  No closed form for n is claimed.
    """,

    "identities": """
        theta(gamma)       = -(pi/2) gamma/(1+gamma)   [surface deflection]
        arg psi(inf)       = pi/[2(1+gamma)] - pi      [deep asymptote]
        arg psi(inf) - arg psi(0) = -pi/2 (mod 2 pi), every gamma < 1
        delta_gamma        = (K_gamma/f)^(1/(1+gamma)) [depth scale]
        psi(0)             = tau p0/(i f rho)
        A_gamma            = tau (p0/b)/Gamma(1-gamma) [tail amplitude]
        psi(0,t)           = psi(0,inf) P(a, i f t), a = gamma/(1+gamma)
        envelope exponent  = a - 1 = -1/(1+gamma)

        The tail amplitude carries the factor 1/Gamma(1-gamma), which
        vanishes at gamma = 1.  That is the exact mechanism by which the
        algebraic far field disappears in the classical limit and the
        winding becomes unbounded.  The limit gamma -> 1 is therefore
        singular, not continuous.

        Only the modulo-2pi statement is exact.  The NET turning is
        -90 - 360 n degrees, and n is not identically zero: it is 0 up to
        gamma of about 0.90, 1 from about 0.91 to 0.995, and at least 2
        by gamma = 0.999.  Any claim that the spiral makes exactly a
        quarter turn at every gamma < 1 is false, and the divergence of n
        is how the unbounded classical winding is approached.
    """,

    "algorithms": """
        contour   fixed-Talbot inversion of the depth transform, M = 32
        series    arbitrary-precision Mittag-Leffler summation (mpmath)
        PI        product-integration march of the Volterra system

        No two of these share a code path.  The public solution splices
        series below zeta = 0.05 and contour above, because the Talbot
        contour radius r = 2M/(5 zeta) overflows the dynamic range as
        zeta -> 0.  Cross-method agreement is reported on the contour
        branch alone so the splice is not being tested against itself.
    """,

    "units": """
        Every plotted and tabulated quantity is DIMENSIONLESS.  The
        reference case is f = 1, K_gamma = 1, rho = 1, and tau = 1
        aligned with +x, so delta_gamma = 1 by construction.  Scaling
        zeta by delta_gamma, psi by |tau| delta_gamma^gamma/(rho
        K_gamma), t by 1/f and T by |tau| reduces the problem to one that
        depends on gamma alone.

        The closure coefficient K_gamma carries units of m^(1+gamma)
        s^-1, so it is a DIMENSIONALLY DIFFERENT quantity at each gamma.
        Setting K_gamma = 1 therefore does not fix a common physical
        mixing strength across the family, and the absolute depth scale
        delta_gamma = (K_gamma/f)^(1/(1+gamma)) is not comparable between
        different gamma at fixed numerical K_gamma.

        What IS comparable across gamma, because it is independent of
        K_gamma entirely: the surface deflection angle, the deep
        asymptotic direction, the turning, the winding number, the decay
        exponent 1+gamma, the spin-up exponent a, and every ratio plotted
        against zeta/delta_gamma.  These are the results the study
        reports.

        What is NOT comparable across gamma at fixed numerical K_gamma:
        the absolute value of delta_gamma in metres, the absolute surface
        speed, and the absolute magnitude of the stress at a given depth
        in metres.  None of these is reported as a physical number
        anywhere here.

        Figure 1(a) plots zeta^gamma times the kernel, which removes its
        m^-gamma units and is the only form in which the kernel may be
        compared across gamma.  Figure 1(b) uses a synthetic test profile
        whose scale is arbitrary; only its slope carries meaning.
    """,

    "symbols": """
        zeta        depth below the surface, positive downward
        psi         complex velocity u + i v
        T           complex stress T_x + i T_y
        tau         surface wind stress
        gamma       order of the Scott-Blair closure, 0 < gamma <= 1
        mu          1 + gamma, order of the governing equation
        K_gamma     closure coefficient, m^(1+gamma) s^-1, gamma dependent
        b           i f / K_gamma
        p0          b^(1/mu), the right-half-plane characteristic root
        delta_gamma (K_gamma/f)^(1/mu), generalized Ekman depth
        a           gamma/(1+gamma), spin-up exponent
        n           winding number of the velocity vector
        M           Talbot contour nodes (32)
        h           product-integration grid spacing
    """,

    "corrected_claim": """
        The quarter-turn result was first stated as: the total turning of
        psi from surface to bottom is exactly -90 degrees for every
        gamma < 1.  That is false.  Only the modulo-2pi statement
        survives.  Measuring the unwrapped turning out to zeta = 1e4
        gives -450 degrees at gamma = 0.95 and -810 degrees at
        gamma = 0.999, that is one and two complete extra revolutions.
        The corrected statement is that the deep direction is a quarter
        turn from the surface direction modulo 2 pi at every gamma, while
        the winding number n(gamma) is a non-decreasing integer that
        diverges as gamma -> 1.  The earlier version was checked only to
        zeta = 1e3 and only up to gamma = 0.9, which is below the first
        threshold.
    """,

    "negative_result": """
        The second root of p^mu = b enters the principal sheet at
        gamma = 1/2.  That crossing was expected to mark a qualitative
        change in the approach to the deep asymptote.  It does not.  The
        overshoot beyond the quarter turn grows smoothly through
        gamma = 1/2, measuring 3.0 degrees at gamma = 0.4 and 8.2 degrees
        at gamma = 0.5, with no discontinuity in value or slope.  The
        crossing has no reported signature.
    """,

    "assumption": """
        The Caputo derivative in the closure is based at the surface, so
        the stress at depth depends on the shear above it and not below.
        That is a modeling choice justified by the direction of momentum
        input, not a derived property.  A Riesz (symmetric) closure would
        forfeit the Laplace-transform solution and change the surface
        condition; what it gives instead is not addressed here.
    """,

    "not_addressed": """
        K_gamma is a constant of the closure and is not derived from any
        eddy statistics.  No calibration against observations or
        large-eddy simulation is attempted, and none of the numbers here
        is a measurement.  The problem is linear, horizontally
        homogeneous, and unstratified.
    """,

    "notes_purpose": """
        Figures carry no in-panel annotation.  Every numerical value that
        would otherwise have been printed inside an axes is recorded
        here, keyed to the figure it belongs to.  All plotted quantities
        are dimensionless; see parameters.txt for the scaling and for
        what may and may not be compared across gamma.
    """,

    "fig0": """
        Diagram only.  The filled profile is the kernel
        (zeta - xi)^-gamma over 0 < xi < zeta, drawn at gamma = 0.6 for
        legibility; the grey bar on the right is the local limit
        gamma = 1, where the same weight collapses onto the evaluation
        depth.  No numerical values are plotted.
    """,

    "fig1": """
        The fitted exponent is taken over the lowest decade of the range;
        the residual of a few times 1e-4 is the next-order term, not a
        failure of the scaling.  The magnitude tends to zero at every
        gamma < 1, which is the statement that the surface stress cannot
        be imposed on psi.  The abscissa of panel (b) is in the arbitrary
        units of the synthetic test profile: only the slope is meaningful.
    """,

    "fig2": """
        The radial coordinate is compressed as |psi/psi_0|^(1/3) so that
        the classical winding stays visible; angles are undistorted by
        that map.  The turning column is modulo 360 degrees, and the
        winding number is zero at all four values of gamma shown.
    """,

    "fig3": """
        Overshoot is the largest excursion beyond the quarter turn.  It
        grows smoothly with gamma and shows no discontinuity at
        gamma = 1/2, where a second root of p^(1+gamma) = b enters the
        principal sheet.

        Once the overshoot exceeds 360 degrees the velocity vector has
        made a complete extra revolution and the NET turning is -450
        rather than -90.  That happens between gamma = 0.90 and 0.91.
        Winding numbers are tabulated in closed_forms.txt.  The
        gamma = 0.95 curve in figure 3(b) is included to show this.

        The narrow spike on the gamma = 0.9 curve near zeta = 7 is a
        genuine near-zero of psi, where the argument is poorly
        conditioned, and not a numerical artifact.
    """,

    "fig4": """
        The compensated speed zeta^(1+gamma)|psi| converges onto the
        closed-form amplitude gamma |A_gamma|/(f rho) with no fitted
        constant.  At gamma = 1 the amplitude is identically zero and the
        decay is exponential.
    """,

    "fig5": """
        The contour comparison is valid only for f t <= 10 at M = 32,
        beyond which the Talbot contour stops enclosing the branch point
        at sigma = -i f.  The quadrature comparison is valid at any t.
    """,

    "fig6": """
        The contour rule reaches 4e-12 near M = 20 to 24 and degrades
        beyond it as roundoff is amplified by exp(2M/5); M = 32 is used
        throughout at a residual near 1e-10.  Transport residuals are
        relative to |tau|/(rho f) and are reported in verification.txt.
        The product-integration value falls with gamma because its
        flux-form update telescopes exactly in the classical limit; the
        contour value is set by the quadrature of a profile with a
        zeta^gamma cusp at the surface.
    """,
}
