"""Generate the plain-text computation reports.

All narrative text lives in :mod:`bayu.reporttext`; this module computes the
numbers and assembles them into tables.  Every quantity written here is
dimensionless, in the reference scaling documented in parameters.txt.
"""

import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
from scipy.special import erf
from scipy.special import gamma as gammafn

from bayu.core import (Params, asymptotic_angle, characteristic_root,
                       deflection_angle, surface_velocity, tail_amplitude,
                       turning_mod_2pi)
from bayu.io_utils import write_report
from bayu.mittagleffler import velocity_series
from bayu.reporttext import block
from bayu.solution import stress, velocity, velocity_contour
from bayu.unsteady import (surface_spinup, surface_spinup_quad,
                           surface_spinup_talbot,
                           transient_envelope_exponent)
from bayu.volterra import solve_volterra

GAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GAMS_WIND = [0.2, 0.4, 0.6, 0.8, 0.9, 0.91, 0.95, 0.99, 0.995, 0.999]
GAMS_ORDER = [0.3, 0.5, 0.7, 0.9]
NS_ORDER = [200, 400, 800, 1600, 3200]
Z_WIND = np.logspace(-4, 4.0, 1400)
T_CONTOUR = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
T_QUAD = np.array([0.1, 1.0, 10.0, 100.0, 400.0])


def rule(width):
    """Horizontal rule of the given width, indented two spaces."""
    return "  " + "-" * width


def turning(prm, depths):
    """Cumulative turning of psi in degrees, unwrapped from the surface."""
    psi = velocity(prm, depths)
    return np.degrees(np.unwrap(np.angle(psi)) - np.angle(psi[0]))


def closed_form_rows():
    """Table of the closed-form angles and exponents against gamma."""
    rows = ["  gamma | arg p0 (deg) |  theta (deg) | arg psi(inf) |"
            " turn mod 360 |  a = g/(1+g)", rule(88)]
    for g in GAMS:
        prm = Params(gamma=g)
        turn = "n/a" if g >= 1.0 else f"{turning_mod_2pi(g):12.2f}"
        rows.append(
            f"  {g:5.2f} |"
            f" {np.degrees(np.angle(characteristic_root(prm))):12.7f} |"
            f" {deflection_angle(g):12.7f} |"
            f" {asymptotic_angle(g):12.7f} |"
            f" {turn:>12s} | {g / (1 + g):13.10f}")
    return rows


def winding_rows():
    """Table of measured net turning and winding number against gamma."""
    rows = ["  gamma | net turning (deg) |  n  | residual from -90-360n",
            rule(66)]
    for g in GAMS_WIND:
        turn = turning(Params(gamma=g), Z_WIND)
        n = int(round((-turn[-1] - 90.0) / 360.0))
        rows.append(f"  {g:5.3f} | {turn[-1]:17.4f} | {n:3d} |"
                    f" {turn[-1] + 90.0 + 360.0 * n:+22.4f}")
    return rows + [""] + block("winding_note")


def volterra_orders():
    """Observed convergence order of the product-integration march."""
    out = []
    for g in GAMS_ORDER:
        prm = Params(gamma=g)
        err = []
        for n in NS_ORDER:
            depths, psi, _ = solve_volterra(prm, 4.0, n)
            err.append(np.max(np.abs(psi[1:] - velocity(prm, depths[1:]))))
        step = 4.0 / np.array(NS_ORDER, dtype=float)
        out.append((g, np.polyfit(np.log(step), np.log(err), 1)[0]))
    return out


def transport_residual_contour():
    """Largest relative residual of the transport invariant, contour path."""
    worst = 0.0
    nodes = np.linspace(0.0, 40.0 ** (1.0 / 3.0), 6001)
    depths = nodes ** 3
    for g in GAMS:
        prm = Params(gamma=g)
        total = (np.trapezoid(3.0 * nodes ** 2 * velocity(prm, depths), nodes)
                 + stress(prm, 40.0)[0] / (1j * prm.f * prm.rho))
        ref = abs(prm.tau) / (prm.rho * prm.f)
        worst = max(worst, abs(total + 1j * prm.tau / (prm.rho * prm.f)) / ref)
    return worst


def transport_residual_pi():
    """Largest relative residual of the transport invariant, PI path."""
    worst = 0.0
    for g in GAMS:
        prm = Params(gamma=g)
        depths, psi, stresses = solve_volterra(prm, 40.0, 8000)
        total = (np.trapezoid(psi, depths)
                 + stresses[-1] / (1j * prm.f * prm.rho))
        ref = abs(prm.tau) / (prm.rho * prm.f)
        worst = max(worst, abs(total + 1j * prm.tau / (prm.rho * prm.f)) / ref)
    return worst


def verification_lines():
    """Measured residuals of every closed form, one line each."""
    lines = []
    ang = max(abs(np.degrees(np.angle(
        velocity_series(Params(gamma=g), 1e-200)[0])) - deflection_angle(g))
        for g in GAMS)
    lines.append("  deflection angle, series at zeta = 1e-200 ......... "
                 f"{ang:.3e} deg")

    probe = np.array([0.25, 1.0, 4.0, 16.0])
    cross = max(np.max(np.abs(velocity_contour(Params(gamma=g), probe)
                              - velocity_series(Params(gamma=g), probe)))
                for g in GAMS)
    lines.append("  contour vs series, 0.25 < zeta < 16 .............. "
                 f"{cross:.3e}")

    deep = np.array([1e5])
    tail = max(abs(np.abs(velocity(Params(gamma=g), deep))[0]
                   * deep[0] ** (1 + g)
                   / (g * abs(tail_amplitude(Params(gamma=g)))) - 1.0)
               for g in GAMS[:-1])
    lines.append("  tail amplitude at zeta = 1e5, relative ........... "
                 f"{tail:.3e}")

    lines.append("  product-integration order, observed vs predicted:")
    for g, order in volterra_orders():
        lines.append(f"      gamma = {g:.1f} ............................. "
                     f"{order:.4f}  vs {min(2.0, 1 + g):.4f}")

    lines.append("  transport invariant, contour, relative ........... "
                 f"{transport_residual_contour():.3e}")
    lines.append("  transport invariant, PI, relative ................ "
                 f"{transport_residual_pi():.3e}")

    spin_ct = max(np.max(np.abs(surface_spinup(Params(gamma=g), T_CONTOUR)
                                - surface_spinup_talbot(Params(gamma=g),
                                                        T_CONTOUR)))
                  for g in GAMS)
    spin_q = max(np.max(np.abs(surface_spinup(Params(gamma=g), T_QUAD)
                               - surface_spinup_quad(Params(gamma=g), T_QUAD))
                        / np.abs(surface_spinup(Params(gamma=g), T_QUAD)))
                 for g in GAMS)
    lines.append("  spin-up vs contour inversion, f t <= 10 .......... "
                 f"{spin_ct:.3e}")
    lines.append("  spin-up vs quadrature, f t <= 400, relative ...... "
                 f"{spin_q:.3e}")

    prm = Params(gamma=1.0)
    classical = np.max(np.abs(surface_spinup(prm, T_QUAD)
                              - surface_velocity(prm)
                              * erf(np.sqrt(1j * T_QUAD))))
    lines.append("  classical limit vs erf(sqrt(i f t)) .............. "
                 f"{classical:.3e}")
    return lines


def notes_fig1():
    """Fitted Caputo degeneracy exponents against the prediction."""
    rows = ["  gamma | fitted slope | predicted 1-gamma | |difference|",
            rule(58)]
    depths = np.logspace(-5, -0.3, 22)
    for g in [0.2, 0.4, 0.6, 0.8, 0.95]:
        vals = np.array([abs(_caputo(g, z)) for z in depths])
        slope = np.polyfit(np.log(depths[:12]), np.log(vals[:12]), 1)[0]
        rows.append(f"  {g:5.2f} | {slope:12.6f} | {1 - g:17.6f} |"
                    f" {abs(slope - (1 - g)):12.2e}")
    return rows + [""] + block("fig1")


def _caputo(g, z):
    """Caputo derivative of the synthetic test profile, order g, depth z."""
    from scipy.integrate import quad
    a = 1.0 - g

    def part(deriv):
        def integrand(s):
            return deriv(z * (1.0 - s ** (1.0 / a))) * (z ** a / a)
        return quad(integrand, 0.0, 1.0, limit=200)[0]

    def dre(x):
        return -np.exp(-x) * (np.cos(x) + np.sin(x)) + 0.6 * x

    def dim(x):
        return np.exp(-x) * (2.0 * np.cos(2.0 * x) - np.sin(2.0 * x))

    return (part(dre) + 1j * part(dim)) / gammafn(1.0 - g)


def notes_fig2():
    """Per-panel angles for the hodograph figure."""
    rows = ["  panel | gamma | arg psi(0) | arg psi(inf) | turn mod 360",
            rule(60)]
    for label, g in zip("abcd", [0.25, 0.50, 0.75, 1.00]):
        far = "n/a" if g >= 1 else f"{asymptotic_angle(g):12.6f}"
        turn = "unbounded" if g >= 1 else f"{-90.0:12.2f}"
        rows.append(f"   ({label})  | {g:5.2f} | {deflection_angle(g):10.6f} |"
                    f" {far:>12s} | {turn:>12s}")
    return rows + [""] + block("fig2")


def notes_fig3():
    """Measured angles, net turning, winding, and overshoot."""
    rows = ["  gamma | measured arg psi(0) | closed form |"
            " net turn |  n  | overshoot", rule(74)]
    for g in [0.2, 0.4, 0.5, 0.7, 0.9, 0.95]:
        meas = np.degrees(np.angle(
            velocity_series(Params(gamma=g), 1e-200)[0]))
        turn = turning(Params(gamma=g), Z_WIND)
        n = int(round((-turn[-1] - 90.0) / 360.0))
        rows.append(f"  {g:5.2f} | {meas:19.10f} |"
                    f" {deflection_angle(g):11.6f} | {turn[-1]:8.2f} |"
                    f" {n:3d} | {max(0.0, -turn.min() - 90.0):9.3f}")
    return rows + [""] + block("fig3")


def notes_fig4():
    """Compensated tail amplitudes against the closed form."""
    depths = np.array([1e2, 1e3, 1e4, 1e5])
    rows = ["  gamma |  zeta=1e2    1e3       1e4       1e5    |"
            " closed form | rel. err", rule(74)]
    for g in [0.2, 0.4, 0.6, 0.8]:
        prm = Params(gamma=g)
        comp = np.abs(velocity(prm, depths)) * depths ** (1 + g)
        pred = g * abs(tail_amplitude(prm)) / prm.f
        rows.append(f"  {g:5.2f} | "
                    + "  ".join(f"{x:.6f}" for x in comp)
                    + f" | {pred:11.6f} | {abs(comp[-1] / pred - 1):.2e}")
    return rows + [""] + block("fig4")


def notes_fig5():
    """Spin-up exponents and cross-method residuals."""
    rows = ["  gamma | a = g/(1+g) | envelope a-1 | vs contour |"
            " vs quadrature", rule(68)]
    for g in [0.2, 0.5, 0.8, 1.0]:
        prm = Params(gamma=g)
        d_ct = np.max(np.abs(surface_spinup(prm, T_CONTOUR)
                             - surface_spinup_talbot(prm, T_CONTOUR)))
        d_q = np.max(np.abs(surface_spinup(prm, T_QUAD)
                            - surface_spinup_quad(prm, T_QUAD))
                     / np.abs(surface_spinup(prm, T_QUAD)))
        rows.append(f"  {g:5.2f} | {g / (1 + g):11.8f} |"
                    f" {transient_envelope_exponent(g):12.8f} |"
                    f" {d_ct:10.2e} | {d_q:13.2e}")
    return rows + [""] + block("fig5")


def notes_fig6():
    """Observed convergence orders for the verification figure."""
    rows = ["  gamma | observed order | predicted min(2, 1+gamma)", rule(54)]
    for g, order in volterra_orders():
        rows.append(f"  {g:5.2f} | {order:14.4f} | {min(2.0, 1 + g):25.4f}")
    return rows + [""] + block("fig6")


def main():
    """Write every report and print the paths."""
    paths = [
        write_report(
            "closed_forms",
            "Closed-form quantities of the nonlocal Ekman layer",
            [("characteristic root, angles, and the transient exponent",
              closed_form_rows()),
             ("measured winding number", winding_rows()),
             ("identities", block("identities"))]),
        write_report(
            "verification", "Numerical verification",
            [("three independent algorithms", block("algorithms")),
             ("measured residuals", verification_lines())]),
        write_report(
            "parameters", "Symbols, units, and reference values",
            [("units, and what is comparable across gamma", block("units")),
             ("symbols", block("symbols"))]),
        write_report(
            "open_items", "Open items and negative results",
            [("a claim that was tested and had to be corrected",
              block("corrected_claim")),
             ("what was tested and did not hold", block("negative_result")),
             ("assumption carried by the whole construction",
              block("assumption")),
             ("not addressed", block("not_addressed"))]),
        write_report(
            "figure_notes", "Figure notes and tabulated annotations",
            [("purpose", block("notes_purpose")),
             ("figure 0, schematic of the closure", block("fig0")),
             ("figure 1, kernel and the degeneracy", notes_fig1()),
             ("figure 2, hodographs", notes_fig2()),
             ("figure 3, angles, turning, and winding", notes_fig3()),
             ("figure 4, depth profiles and the algebraic tail",
              notes_fig4()),
             ("figure 5, surface spin-up", notes_fig5()),
             ("figure 6, verification", notes_fig6())]),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
