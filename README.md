# Station-keeping of the James Webb Space Telescope around L2

**A from-scratch numerical study of halo-orbit dynamics in the Sun–Earth CR3BP, and of the corrections needed to keep a spacecraft on it.**

TIPE project (French *classes préparatoires*, MP, theme *Cycles, boucles*) · Python · NumPy / SciPy / Matplotlib

> **Research question:** *How can the position of the JWST be adjusted so that it stays in orbit around the Sun–Earth L2 point?*

---

## Why this problem

The JWST sits near the Sun–Earth **L2** point, about 1.5 million km from Earth. There the Sun–Earth geometry is nearly constant, which gives a thermally stable environment and continuous communication. But L2 is an **unstable** equilibrium: left alone, the telescope drifts away, so it needs periodic corrections. This project models that instability and computes the corrections that counter it.

## Model

- **CR3BP**: circular restricted three-body problem, in 3D, in the rotating frame centred on the Sun–Earth barycentre, in dimensionless units with μ ≈ 3.04 × 10⁻⁶ (JPL value).
- The JWST (~6.5 t) is treated as a massless test particle.
- Not modelled: the Moon, solar radiation pressure, seasonal variations of the Earth–Sun distance, other planets. These are discussed as limits below.

## Method

Everything below is implemented from scratch: no orbital-mechanics library. (I first tested REBOUND, then dropped it because of its opaque internals and to keep full control over the integration).

1. **Equations of motion and pseudo-potential** U\*, with the Jacobi constant C = 2U\* − v² as the validation invariant.
2. **RK4 integrator** (local error O(h⁵)), used for the trajectory, the reference orbit and the state-transition matrix.
3. **Locating L2**: the collinear point is found from the quintic in γ (Brent's method).
4. **Initial halo orbit**: 3rd-order Richardson approximation.
5. **Differential correction (single shooting)**: Newton iterations on the state-transition matrix, exploiting the xz-plane symmetry, to turn the Richardson seed into a periodic halo orbit. The method was suggested during an exchange with the mathematician Emmanuel Trélat.
6. **Stability analysis**: analytic Jacobian, state-transition matrix integrated as a 42-dimensional augmented system, monodromy matrix, then stable and unstable eigenvectors (left/right pairing, biorthogonal normalisation).
7. **Station-keeping (EVSK, eigenvector-based)**: every 21 days, a velocity impulse cancels the projection of the deviation on the unstable direction. Eigenvectors are propagated along the orbit by the STM, and ΔV is capped at 2 m/s per manoeuvre.
8. **Cross-check in an inertial J2000-like frame**: an N-body (Sun + Earth) simulation with the same integrator, compared through energy conservation.

## Results

| | |
| --- | --- |
| Halo amplitude (out-of-plane) | A_z ≈ 418 000 km (0.00279 in dimensionless units) |
| Simulated duration | 6 halo revolutions (~1080 days), 10 000 RK4 steps per revolution |
| Integration quality | Jacobi constant conserved to ~10⁻⁸ over the first ~670 days |
| Without correction | The spacecraft leaves the halo region within about twenty days once perturbed (initial 100 km radial offset) |
| With EVSK corrections | Position error kept around **10²–4 × 10² km** for ~650 days, against up to ~10⁸ km without corrections |

<!-- TODO: add total ΔV, number of manoeuvres and halo period from the printed [Station-Keeping] summary,
     compared with the ~2.5 m/s/year quoted for the real JWST. -->

## Limitations and next steps

- **The reference orbit itself drifts** beyond about 3 halo periods: the single-shooting correction converges over a few revolutions but is not stable on longer horizons. This is why the station-keeping error jumps after ~650 days, and it is a limit of the reference orbit, not of the correction logic.
- A first station-keeping attempt failed before EVSK worked, because the available NASA documents were not sufficient to reproduce the method.
- **Natural next step: multiple shooting**, which is far better conditioned for long unstable orbits (see Chupin's thesis in the references).
- Model extensions: Moon, solar radiation pressure, seasonal variations, other planets.

## Project structure

```
TIPE-LAGRANGE/
           
├── src/
│   ├── main.py          # runs CR3BP, station-keeping and inertial simulations + plots
│   ├── core/            # Body, System, RK4 integrator
│   ├── cr3bp/           # equations, Lagrange points, Richardson + STM/monodromy, station-keeping
│   ├── inertial/        # inertial N-body simulation and forces
│   └── visualization/   # trajectory, velocity, energy/Jacobi and station-keeping plots
├── plots/               # figures
└── docs/                # MCOT and oral presentation (in French)
```

## Running it

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
python main.py
```

Figures are written under `plots/`.

## Team and credits

Group project with **Joe Bimont**.

- **Victor Averseng**: all programming: numerical methods, simulations, station-keeping algorithm, validation and visualisation.
- **Joe Bimont**: physical theory, derivations and proofs (presented in his own oral).

## Documents

The MCOT and the oral presentation are in [`docs/`](docs/) (French).

## References

1. B. Meyssignac, *Introduction à la Mécanique Céleste*, ISAE-SUPAERO.
2. J. Petersen, *L2 station keeping maneuver strategy for the James Webb Space Telescope*, NASA.
3. M. Menzel et al., *The Design, Verification, and Performance of the James Webb Space Telescope*, NASA.
4. W. Yu, *James Webb Space Telescope Trajectory Design Overview*, NASA.
5. J. Brown, J. Petersen, J. Villac, W. Yu, *Seasonal variations of the James Webb Space Telescope orbital dynamics*, NASA.
6. M. Chupin, *Interplanetary transfers with low consumption using the properties of the restricted three body problem*, PhD thesis, Paris 6, 2016.
7. D. L. Richardson, *Analytic construction of periodic orbits about the collinear points*, Celestial Mechanics 22 (1980), 241–253.

## License

MIT
