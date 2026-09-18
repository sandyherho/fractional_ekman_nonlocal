# Supplementary Scripts: **An exactly solvable Ekman layer with a fractional-order stress closure**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.10-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![mpmath](https://img.shields.io/badge/mpmath-%E2%89%A51.3-4B6C8C?style=flat-square)](https://mpmath.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%E2%89%A53.7-11557C?style=flat-square)](https://matplotlib.org)
[![pycodestyle](https://img.shields.io/badge/pycodestyle-clean-1E7B7B?style=flat-square)](https://peps.python.org/pep-0008/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-clean-1E7B7B?style=flat-square)](https://peps.python.org/pep-0257/)
[![License: MIT](https://img.shields.io/badge/License-MIT-A31F34?style=flat-square)](LICENSE)
[![No external data](https://img.shields.io/badge/data-none%20required-7D8CA3?style=flat-square)](#)

A wind-driven boundary layer closed by a fractional-order flux-gradient law
instead of a local eddy viscosity. Analysis only: no model output, no
observational input.

<div align="center">
  <img src="outputs/figures/fig00_schematic.png" alt="nonlocal Ekman closure" width="400" />
</div>

## Model

On an $f$-plane, with $\zeta=-z$ the depth, $\psi=u+iv$ and $T=T_x+iT_y$,

$$i f \rho\,\psi = -\partial_\zeta T, \qquad T(0)=\tau, \qquad T(\infty)=0,$$

which alone gives $\int_0^\infty\psi\,d\zeta=-i\tau/(\rho f)$ for any closure.
The local law is replaced by a Scott-Blair law of order $\gamma$,

$$T = -\rho K_\gamma\,{}^{C}\!D^{\gamma}_{\zeta}\psi, \qquad 0<\gamma\leq1,$$

so the stress at depth depends on a power-law-weighted history of the shear
above. Setting $\gamma=1$ recovers Ekman exactly.

## Results

**Posing it on $\psi$ fails.** A Caputo derivative based at the surface gives
${}^{C}\!D^{\gamma}_{\zeta}\psi = O(\zeta^{1-\gamma})\to0$, forcing $T(0)=0$, so
the wind stress cannot be imposed. Riemann-Liouville diverges as
$\zeta^{-\gamma}$ instead. The problem must be posed on the stress.

**Posed on $T$, it is exactly solvable.** With $\mu=1+\gamma$, $b=if/K_\gamma$,
and $p_0=b^{1/\mu}$ the right-half-plane root that $T(\infty)=0$ cancels,

$$\hat T(p) = \tau\,\frac{p^{\gamma-1}(p-p_{0})}{p^{\mu}-b}, \qquad
T(\zeta)=\tau\left[E_{\mu,1}(b\zeta^{\mu})-p_{0}\zeta E_{\mu,2}(b\zeta^{\mu})\right].$$

| quantity | closed form |
| --- | --- |
| surface velocity | $\psi(0)=\tau p_{0}/(if\rho)$ |
| deflection angle | $\theta=-\tfrac{\pi}{2}\gamma/(1+\gamma)$, $-45^\circ$ at $\gamma=1$ |
| depth scale | $\delta_\gamma=(K_\gamma/f)^{1/(1+\gamma)}$ |
| far field | $T\sim A_\gamma\zeta^{-\gamma}$, $A_\gamma=\tau(p_0/b)/\Gamma(1-\gamma)$ |
| deep direction | $\arg\psi(\infty)-\arg\psi(0)=-\tfrac{\pi}{2}\pmod{2\pi}$ |
| spin-up | $\psi(0,t)=\psi(0,\infty)P(a,ift)$, $a=\gamma/(1+\gamma)$ |

$1/\Gamma(1-\gamma)$ vanishes at $\gamma=1$, removing the algebraic tail and
making the classical limit singular. The net turning is
$-\tfrac{\pi}{2}-2\pi n$, with $n$ measured: $0$ up to $\gamma\simeq0.90$, $1$ to
$\simeq0.995$, $\geq2$ by $0.999$. At $\gamma=1$ the spin-up reduces to
$\mathrm{erf}(\sqrt{ift})$; its transient envelope decays as
$(ft)^{-1/(1+\gamma)}$, never exponentially.

## Numerics

Three algorithms, no shared code path: fixed-Talbot contour inversion
(rewritten for complex-valued inverses), arbitrary-precision Mittag-Leffler
summation, and product-integration marching of the equivalent Volterra system.
The public solution splices series below $\zeta=0.05$ and contour above;
cross-method agreement is reported on the contour branch alone.

| check | result |
| --- | --- |
| deflection angle vs closed form | $1.1\times10^{-14}$ deg |
| contour vs series, $0.25<\zeta<16$ | $6.9\times10^{-10}$ |
| tail amplitude at $\zeta=10^{5}$, relative | $3.2\times10^{-5}$ |
| product-integration order, $\gamma=0.3\ldots0.9$ | 1.294, 1.483, 1.658, 1.803 |
| predicted $\min(2,1+\gamma)$ | 1.300, 1.500, 1.700, 1.900 |
| transport invariant, both paths, relative | $<10^{-9}$ |
| spin-up vs quadrature, $ft\leq400$, relative | $1.2\times10^{-11}$ |
| classical limit vs $\mathrm{erf}(\sqrt{ift})$ | $1.6\times10^{-15}$ |

## Units

Everything is dimensionless. The reference case is $f=K_\gamma=\rho=\tau=1$, so
$\delta_\gamma=1$ and the problem depends on $\gamma$ alone. $K_\gamma$ carries
units $\mathrm{m}^{1+\gamma}\mathrm{s}^{-1}$ and is dimensionally different at
each $\gamma$, so $K_\gamma=1$ does not fix a common mixing strength.
Comparable across $\gamma$: the angles, the winding number, the exponents
$1+\gamma$ and $a$, and every ratio against $\zeta/\delta_\gamma$. Not
comparable, and not reported: $\delta_\gamma$ in metres and absolute speeds.

## Run

```bash
pip install -r requirements.txt   # or: pip install -e .
python scripts/run_all.py
```

About three minutes. Figure scripts run standalone. Lint with
`pycodestyle bayu/ scripts/` and `pydocstyle bayu/ scripts/`.

## Layout

```
bayu/
  core.py           parameters, characteristic root, closed forms
  transform.py      fixed-Talbot inversion, complex-valued, full contour
  mittagleffler.py  arbitrary-precision E_{mu,nu} and the series solution
  solution.py       depth transform, spliced stress and velocity, far field
  volterra.py       product-integration march of the Volterra system
  unsteady.py       spin-up: closed form, contour inversion, quadrature
  plotting.py       style, palette, PDF and 600 dpi PNG export
  io_utils.py       CSV and plain-text report writers
  reporttext.py     report prose, separated from the numbers
scripts/
  _bootstrap.py     puts the repository root on sys.path
  fig00 .. fig06    one script per figure
  make_reports.py   plain-text reports
  run_all.py        regenerate everything
outputs/
  figures/          vector PDF and 600 dpi PNG
  data/             every panel as CSV
  reports/          closed forms, verification, parameters and units,
                    figure notes, open items
```

Outputs are committed so the figures and numbers can be inspected without
running anything. Figures carry no titles and no in-panel annotation; panel
labels and legends sit outside the axes, and tabulated values are in
`outputs/reports/figure_notes.txt`.

## Limitations

The Caputo derivative is based at the surface, so the stress at depth depends on
the shear above and not below. That is a modeling choice justified by the
direction of momentum input, not a derived property.

$K_\gamma$ is not derived from eddy statistics, nothing is calibrated, and no
number here is a measurement. The deflection-angle deficit is offered as a
mechanism, not an explanation established against data.

The winding number is measured, not derived. Its thresholds are resolved only to
the $\gamma$ grid used, and the rate at which it diverges as $\gamma\to1$ is not
established.

The crossing of a second root into the principal sheet at $\gamma=1/2$ was
expected to leave a signature in the approach to the deep asymptote. It does
not.

The problem is linear, horizontally homogeneous, unstratified, and forced by a
single switch-on.

## Authors

Sandy H. S. Herho, Iwan P. Anwar, Rusmawan Suwarman, Deny J. Puradimaja,
Dasapta E. Irawan

## License

MIT. See [LICENSE](LICENSE).