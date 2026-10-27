# Diabatic and adiabatic propagation in the Sun

`production_mixing` returns the vacuum mass-state fractions `c_i` of solar neutrinos, averaged over a production region. A fast adiabaticity check picks the adiabatic path or integrates the evolution.

## Adiabatic path: the convolution trick (`method="convolution"`)

With adiabatic propagation, a neutrino produced as `nu_alpha` at radius `r` keeps the populations
`|<k(r)|nu_alpha>|^2` of the matter eigenstates, and level `k` leaves the Sun as mass state `k`. When the matter term is `n_e` times a fixed matrix, the populations depend only on `A = 2E V_CC`, so the production average is an FFT correlation of the production kernel with populations tabulated once on a `ln A` grid: 1e-8 from the sum, at a cost independent of the number of energies. Steriles with the NC term (`rho_yn`) use the weighted sum, `method="sum"`.

## Non-adiabatic path: quadratic approximation solver (`method="quadratic"`)

Integrate `i dpsi/dr = H(r) psi` in the flavour basis.

**Each step.** Within a step of length `h`, `H` is expanded about the midpoint as `H_m + h1 s + h2 s^2/2`, with `h1` and `h2` from the left, mid and right nodes.

- `H_m` is diagonalised and taken exactly.
- The rest is taken to first Magnus order in the interaction picture of `H_m`, through the moments `M_p = int s^p e^{i theta s} ds` for `p = 1, 2`.
- The step propagator is `e^{-i H_m h/2} Q exp(-i(h1 M1 + h2 M2/2)) Q^dag e^{-i H_m h/2}`.

The fast oscillation phases (core oscillation length ~65 km) are exact. The step density follows the slope of the potential, `sqrt(||dH/dr||_traceless / eps)`, with `eps = 0.01`.

**Grid.** It is built from the profile and the CC/NC matrices only, not from NSI, `E` or the oscillation parameters. Shapes are static, so the solver is jit-able and differentiable in those.

**Production region.** Production radii are step edges; one reverse associative scan covers them all.

**Reverse mode.** `jax.checkpoint` + `lax.map` over energies: ~0.3 GB flat, ~1.3x the time.

**Accuracy and cost.**
- Accuracy is ~1e-5 against thin layers, from adiabatic to sudden crossings: 40 random 3nu/NSI/3+1 points, worst 1.1e-5.
- Value: 11–19 ms per energy on one core. Value + gradient: ~35–50 ms per energy.
- `method="exact"` is the reference: 200k constant-density layers, same method as the atmospheric osc Earth propagation, ~0.4 s per energy.
- The profile and weights must be concrete. There are no gradients in the solar model.

## Adiabaticity check

The leading error of the adiabatic fractions is first order in `1/gamma` and sits at the production point `r0`. It is removed by starting the neutrino in the first-order superadiabatic state: with `K_jk = i A_jk / (lambda_j - lambda_k)`, `A_jk = <j(r0)|d/dr|k(r0)>` and `b = <k(r0)|nu_alpha>`, the populations are `|((1 - K) b)_j|^2`, kept to first order so they still sum to 1. It costs nothing (eigenvectors and `dH/dr` at the production radii), improves `method="sum"` 2–3x (median), and the check below needs it.

**Criterion:** Compute the smallest adiabaticity
`gamma = (lambda_k - lambda_j)^2 / (2 |<k|dH/dr|j>|)` over pairs of levels in the same block, on 20 nodes uniform in `r` from the innermost production radius to the surface. Also compute the smallest overlap `|<k(r_i)|k(r_{i+1})>|` between neighbouring nodes, which catches crossings narrower than the grid.

- **Accept** the corrected adiabatic fractions if `gamma_min >= 1000` and every overlap is `>= 0.9`.
- **Otherwise** use `method="quadratic"`.

**Robustness:**
- To first order (integration by parts) the error is at most the total variation of `q = 1/(2 gamma)` along the path, so `<= m / gamma_min` for `m` resonances: 1e-4 guaranteed only for `gamma_min >= 1e4`.
- The measured envelope is `error <= 0.06 / gamma_min`; the threshold 1000 is empirical, set by single production radii (worst accepted 1.7e-5). Over production regions the worst accepted point is 3.5e-7. 3000 census points (SF-III): no non-adiabatic point accepted, 11% of adiabatic ones sent to `quadratic`.

**Cost.** `solar.is_adiabatic`: ~3 ms for 8 energies, 6–8 ms for 100.
