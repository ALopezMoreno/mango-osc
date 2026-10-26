"""Solar matter profile and adiabatic MSW evolution.

Provides a standard-solar-model electron-density profile (loaded from a BS05 or a
B23/SF-III table, or an analytic exponential approximation) and the **adiabatic** mass-state
composition of a neutrino produced inside the Sun -- the mechanism behind the LMA
solar solution. Idea adapted from the nu-waves library. :func:`production_mixing`
averages that composition over a source's production region, and
:func:`probability_solar` turns it into probabilities by day and at night. Where
level crossings are not adiabatic, ``method="quadratic"`` integrates the evolution.

In the adiabatic regime a neutrino produced as flavor ``alpha`` populates the
instantaneous matter eigenstates with fixed weights ``w_k = |<nu_k^m|nu_alpha>|^2``
(no level hopping); the observable vacuum mass-state fractions then evolve only
because the matter eigenstates rotate as the density drops:

    F_i(r) = sum_k w_k |<nu_i^vac | nu_k^m(r)>|^2 .
"""

from __future__ import annotations

import functools
import warnings
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

from . import constants as C
from .hamiltonian import matter_hamiltonian
from .oscillator import _check_flavor_index, _is_concrete, _n_active, _nsi_matrix, n_flavors, probability_earth

R_SUN_KM = 695700.0

# ``rho_yn = rho (1 - Y_e)`` is the neutron density in the units of ``rho_ye``, or None; with it, sterile
# flavors feel the neutral-current term.
SolarProfile = namedtuple("SolarProfile", ["r_over_rsun", "rho_ye", "R_sun_km", "rho_yn"], defaults=(None,))


def load_bs05(path):
    """Load a BS05(-AGS,OP) standard solar model table.

    Returns a :class:`SolarProfile` with ``rho_ye = rho * Y_e`` and
    ``rho_yn = rho * (1 - Y_e)`` (g/cm^3), using ``Y_e = (1 + X)/2`` from the
    hydrogen mass fraction X (column 7); radius is column 2 (r/R_sun), density
    column 4.
    """
    rows = []
    started = False
    with open(path) as f:
        for line in f:
            if not started:
                if line.strip().startswith("The Table begins"):
                    started = True
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            rows.append(vals)
    arr = np.array(rows)
    r = arr[:, 1]
    rho = arr[:, 3]
    X = arr[:, 6]
    ye = 0.5 * (1.0 + X)
    return SolarProfile(r_over_rsun=r, rho_ye=rho * ye, R_sun_km=R_SUN_KM, rho_yn=rho * (1.0 - ye))


# (column, Z, A) of the SF-III species; standard atomic weights from Ne on.
_SF3_SPECIES = (("H1", 1, 1), ("He4", 2, 4), ("He3", 2, 3), ("C12", 6, 12), ("C13", 6, 13), ("N14", 7, 14),
                ("N15", 7, 15), ("O16", 8, 16), ("O17", 8, 17), ("O18", 8, 18), ("Ne", 10, 20.180),
                ("Na", 11, 22.990), ("Mg", 12, 24.305), ("Al", 13, 26.982), ("Si", 14, 28.085), ("P", 15, 30.974),
                ("S", 16, 32.06), ("Cl", 17, 35.45), ("Ar", 18, 39.948), ("K", 19, 39.098), ("Ca", 20, 40.078),
                ("Sc", 21, 44.956), ("Ti", 22, 47.867), ("V", 23, 50.942), ("Cr", 24, 51.996), ("Mn", 25, 54.938),
                ("Fe", 26, 55.845), ("Co", 27, 58.933), ("Ni", 28, 58.693))


def _sf3_columns(path):
    with open(path) as f:
        names = f.readline().strip().lstrip("#").split()
    return dict(zip(names, np.loadtxt(path).T))


def load_sf3(path):
    """Load a ``struct+nu_SF3_*.dat`` standard solar model table (Herrera & Serenelli, B23/SF-III).

    The densities count all electrons and neutrons of the mass fractions, as the
    coherent potential does; the table's ``log_ne`` counts free electrons only.
    """
    col = _sf3_columns(path)
    rho = 10.0 ** col["logRho"]
    ne = rho * sum(col[s] * z / m for s, z, m in _SF3_SPECIES)
    nn = rho * sum(col[s] * (m - z) / m for s, z, m in _SF3_SPECIES)
    return SolarProfile(r_over_rsun=col["R_sun"], rho_ye=ne, R_sun_km=10.0 ** col["logR"][-1] / 1e5, rho_yn=nn)


def load_sf3_production(path, source="8B"):
    """Production weights of ``source`` (e.g. ``"8B"``) on the rows of :func:`load_sf3`, summing to 1."""
    col = _sf3_columns(path)
    w = np.clip(col[f"nu_{source}"], 0.0, None) * np.gradient(col["R_sun"])   # clip spline noise near r = 0
    return w / w.sum()


def exponential_profile(ne0_over_NA=245.0, scale=10.54, n=400):
    """Bahcall exponential approximation ``n_e = ne0 * exp(-scale * r/R_sun)``.

    ``ne0_over_NA`` is the central electron density in units of N_A per cm^3, so
    ``rho_ye = n_e / N_A`` (g/cm^3-equivalent) ``= ne0_over_NA * exp(...)``.
    """
    r = np.linspace(0.0, 1.0, n)
    rho_ye = ne0_over_NA * np.exp(-scale * r)
    return SolarProfile(r_over_rsun=r, rho_ye=rho_ye, R_sun_km=R_SUN_KM)


def potential_eV(profile, r_km):
    """Matter potential V (eV) at radius ``r_km`` by interpolating the profile."""
    x = jnp.asarray(r_km) / profile.R_sun_km
    rho_ye = jnp.interp(x, jnp.asarray(profile.r_over_rsun),
                        jnp.asarray(profile.rho_ye))
    return C.matter_potential_eV(rho_ye, 1.0)


def _potentials_eV(profile, r_over_rsun):
    """``(V_CC, V_NC)`` (eV) at ``r_over_rsun``; ``V_NC`` is zero without ``rho_yn``."""
    x, grid = jnp.asarray(r_over_rsun), jnp.asarray(profile.r_over_rsun)
    v_cc = C.matter_potential_eV(jnp.interp(x, grid, jnp.asarray(profile.rho_ye)), 1.0)
    if profile.rho_yn is None:
        return v_cc, jnp.zeros_like(v_cc)
    return v_cc, 0.5 * C.matter_potential_eV(jnp.interp(x, grid, jnp.asarray(profile.rho_yn)), 1.0)


def _blocks(u, msq, n_active, nsi, neutrons):
    """Block label per mass state, and the mass states in (block, m^2) order.

    ``H(r) = H_0 + sum_j V_j(r) P_j`` with ``H_0`` diagonal in the mass basis, so mass states share a block
    when some ``U^dagger P_j U`` links them, directly or not. Levels of different blocks may cross.
    """
    n = msq.shape[0]
    cc = jnp.zeros((n, n), jnp.complex128).at[0, 0].set(1.0)
    if nsi is not None:
        cc = cc.at[:n_active, :n_active].add(jnp.asarray(nsi, jnp.complex128))
    ops = [cc]
    if neutrons and n_active < n:
        ops.append(jnp.diag((jnp.arange(n) >= n_active).astype(jnp.complex128)))
    link = jnp.eye(n, dtype=bool)
    for p in ops:
        link = link | (jnp.abs(jnp.conj(u).T @ p @ u) > 0)
    for _ in range(max(1, int(np.ceil(np.log2(n))))):
        link = (link.astype(jnp.int32) @ link.astype(jnp.int32)) > 0
    first = jnp.argmax(link, axis=1)                       # lowest mass index in each state's block
    label = jnp.cumsum(jnp.zeros(n, jnp.int32).at[first].set(1)).take(first) - 1   # blocks numbered 0, 1, ...
    return label, jnp.lexsort((msq, label))


def _block_shift(u, msq, label, energy_eV, v_max, nsi, anti):
    """A constant per block, above the eigenvalue spread of ``H``, so ``eigh`` returns (block, eigenvalue) order.

    It commutes with ``H``, so the eigenvectors are unchanged; zero for a single block.
    """
    nsi_norm = 0.0 if nsi is None else jnp.sum(jnp.abs(nsi))
    scale = 4.0 * (jnp.ptp(msq) / (2.0 * energy_eV) + v_max * (1.0 + nsi_norm))
    ue = jnp.conj(u) if anti else jnp.asarray(u, jnp.complex128)
    return (ue * (scale * label)) @ jnp.conj(ue).T


def _matter_eigvecs(u, msq, energy_eV, v_cc, v_nc, anti, nsi, n_active, shift=0.0):
    """Matter eigenstates in the flavor basis (columns, ascending eigenvalue; (block, eigenvalue) with ``shift``)."""
    h = matter_hamiltonian(u, msq, energy_eV, v_cc, anti=anti, nsi=nsi, v_nc_eV=v_nc, n_active=n_active)
    return jnp.linalg.eigh(h + shift)[1]


def adiabatic_mass_fractions(params, energy_GeV, profile, r_km,
                             r_emit_km, alpha=0, anti=False):
    """Vacuum mass-state fractions ``F_i(r)`` under adiabatic solar evolution.

    ``r_km`` is a scalar or array of radii (km) at which to evaluate;
    ``r_emit_km`` is the production radius. Sterile flavors feel the neutral-current
    term when the profile carries ``rho_yn``. Returns array of shape
    ``(len(r_km), N)`` for array input, or shape ``(N,)`` for scalar input
    (consistent with the scalar-or-array convention used elsewhere in the API).
    """
    r_km = jnp.asarray(r_km)
    scalar_r = r_km.ndim == 0
    r_km = jnp.atleast_1d(r_km)

    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    energy_eV = energy_GeV * C.GEV_TO_EV
    label, _ = _blocks(u, msq, na, None, profile.rho_yn is not None)
    v_max = jnp.max(jnp.abs(jnp.concatenate(_potentials_eV(profile, profile.r_over_rsun))))
    shift = _block_shift(u, msq, label, energy_eV, v_max, None, anti)

    def eigvecs(r):
        # matter_hamiltonian applies U -> U* and V -> -V itself for antineutrinos.
        v_cc, v_nc = _potentials_eV(profile, jnp.asarray(r) / profile.R_sun_km)
        return _matter_eigvecs(u, msq, energy_eV, v_cc, v_nc, anti, None, na, shift)

    # Production weights w_k = |<nu_k^m(r_emit) | nu_alpha>|^2.
    w = jnp.abs(eigvecs(r_emit_km)[alpha, :]) ** 2  # (N,)

    udag = (u if anti else jnp.conj(u)).T

    def frac_at(r):
        vecs = eigvecs(r)
        a = udag @ vecs  # <nu_i^vac | nu_k^m> = (U^dag V)[i,k]
        return (jnp.abs(a) ** 2) @ w  # F_i = sum_k |a_ik|^2 w_k

    fractions = jax.vmap(frac_at)(r_km)
    return fractions[0] if scalar_r else fractions


_WEIGHT_FLOOR = 1e-14   # production weight, relative to the peak, below which radii are dropped
_DU = 5e-4              # ln(A) grid spacing of the convolution; interpolation error ~ _DU^2
GAMMA_WARN = 10.0       # production_mixing warns below this adiabaticity
_GAMMA_ENERGIES = 8     # energies at which production_mixing checks adiabaticity
_QUADRATIC_EPS = 0.01   # method="quadratic": step size from ||dH/dr|| h^2 ~ eps
_QUADRATIC_MIN_STEPS = 64
_QUADRATIC_GRID_SAMPLES = 4001


def _check_weights(weights, profile):
    """Raise on production weights that cannot describe a production region (concrete input only)."""
    if not _is_concrete(weights):
        return
    w = np.asarray(weights, dtype=float)
    if w.shape != np.shape(profile.r_over_rsun):
        raise ValueError(f"weights must have one entry per profile radius: shape {np.shape(profile.r_over_rsun)}, "
                         f"got {w.shape}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if not w.sum() > 0:
        raise ValueError("weights must not all be zero")


def _kept_rows(weights):
    """Concrete weights and the mask of rows above ``_WEIGHT_FLOOR`` of the peak."""
    w = np.asarray(weights, dtype=float)
    return w, w > _WEIGHT_FLOOR * w.max()


def _all_concrete(*trees):
    return all(_is_concrete(x) for x in jax.tree_util.tree_leaves(trees))


def _dag(x):
    """Conjugate transpose over the last two axes (numpy or JAX)."""
    return x.conj().swapaxes(-1, -2)


def _hamiltonians(params, energy_eV, v_cc, v_nc, anti, nsi):
    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    return jax.vmap(lambda v, vn: matter_hamiltonian(u, msq, energy_eV, v, anti=anti, nsi=nsi, v_nc_eV=vn,
                                                     n_active=na))(v_cc, v_nc)


def adiabaticity(params, energy_GeV, profile, r_min=0.0, anti=False, nsi=None):
    """Smallest adiabaticity parameter on the way out of the Sun from ``r_min`` [R_sun], shape ``([n_E])``.

    ``gamma_ij = (lambda_i - lambda_j)^2 / (2 |<i|dH/dx|j>|)``, minimised over radii
    and pairs of matter eigenstates; at a two-flavor resonance this is the usual
    ``Delta m^2 sin^2(2 theta) / (2 E cos(2 theta) |d ln n_e / dx|)``. The adiabatic
    approximation is good to ~1e-5 above gamma ~ 10 and fails at gamma ~ 1-3.
    """
    r = np.asarray(profile.r_over_rsun, dtype=float)
    sel = np.flatnonzero(r >= r_min)
    x = jnp.asarray(r[sel] * profile.R_sun_km * C.KM_TO_INV_EV)
    v_cc, v_nc = _potentials_eV(profile, jnp.asarray(r[sel]))
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    return _gamma_min(params, jnp.atleast_1d(energy_eV), x, v_cc, v_nc, _nsi_matrix(nsi, _n_active(params)),
                      anti, profile.rho_yn is not None).reshape(energy_eV.shape)


@functools.partial(jax.jit, static_argnames=("anti", "neutrons"))
def _gamma_min(params, energy_eV, x, v_cc, v_nc, nsi, anti, neutrons):
    """Minimum over radii and pairs of matter eigenstates of the adiabaticity parameter, per energy."""
    u, msq = params.pmns(), params.msquared()
    iu = np.triu_indices(msq.shape[0], 1)
    label, _ = _blocks(u, msq, _n_active(params), nsi, neutrons)
    v_max = jnp.max(jnp.abs(jnp.concatenate([v_cc, v_nc])))

    def at_energy(e_eV):
        h = _hamiltonians(params, e_eV, v_cc, v_nc, anti, nsi)
        h = h + _block_shift(u, msq, label, e_eV, v_max, nsi, anti)
        lam, vec = jnp.linalg.eigh(0.5 * (h[1:] + h[:-1]))
        dh = (h[1:] - h[:-1]) / jnp.diff(x)[:, None, None]
        coupling = jnp.abs(jnp.einsum("rai,rab,rbj->rij", jnp.conj(vec), dh, vec))[:, iu[0], iu[1]]
        gap2 = ((lam[:, :, None] - lam[:, None, :]) ** 2)[:, iu[0], iu[1]]
        return jnp.min(jnp.where(coupling > 0, gap2 / (2.0 * jnp.where(coupling > 0, coupling, 1.0)), jnp.inf))

    return jax.vmap(at_energy)(energy_eV)


def production_mixing(params, energy_GeV, profile, weights, alpha=0, anti=False, nsi=None, method="sum",
                      check_adiabaticity=True):
    """Vacuum mass-state fractions of solar neutrinos, averaged over a production region.

    ``weights`` are production fractions per point of ``profile.r_over_rsun`` (already
    integrated over each grid shell, not a density dPhi/dr). Returns ``c[..., i]``,
    shape ``([n_E,] N)``: the fraction produced as ``alpha`` that leaves the Sun as
    mass state ``i``. By day ``P(nu_alpha -> nu_beta) = sum_i c_i |U_{beta i}|^2``; at
    night see :func:`probability_solar`.

    The adiabatic methods (``"sum"``, ``"convolution"``) assume that neutrinos follow
    their matter eigenstates: within each block of the Hamiltonian, the k-th lowest
    leaves as the k-th lightest mass state (a sterile with all ``theta_i4 = 0`` is its
    own block). This fails at narrow avoided crossings, e.g. small active-sterile
    mixing at Delta m^2_41 ~ 1e-5-1e-4 eV^2; on concrete inputs they warn when
    :func:`adiabaticity` drops below ``GAMMA_WARN``, unless
    ``check_adiabaticity=False``.

    ``method="sum"`` diagonalises at every radius and is differentiable in ``params``,
    the profile and the weights. ``method="convolution"`` needs a matter term equal to
    ``n_e`` times a fixed matrix (NSI defined relative to ``n_e`` and antineutrinos
    included; not steriles with ``rho_yn``): the composition then depends only on
    ``A = 2 E V_CC``, and the average is an FFT correlation in ``ln A`` (error ~1e-8).
    It needs concrete energies, profile and weights; gradients in ``params`` are exact.
    ``method="quadratic"`` integrates the evolution without the adiabatic assumption,
    by Magnus steps on a radial grid fixed by the profile (error ~1e-5, ~1e-4 from
    single radii of coarse tables such as BS05; ~10-20 ms per energy). It needs a
    concrete profile and weights and is differentiable in ``params`` and the energies.
    """
    _check_weights(weights, profile)
    u, msq = params.pmns(), params.msquared()
    n, na = msq.shape[0], _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)
    if method == "quadratic":
        return _path_mixing(params, energy_GeV, profile, weights, alpha, anti, nsi_mat, na, method)
    label, mass_order = _blocks(u, msq, na, nsi_mat, profile.rho_yn is not None)

    if method == "convolution":
        if profile.rho_yn is not None and na < n:
            raise ValueError("method='convolution' needs a matter term proportional to n_e; "
                             "sterile flavors with rho_yn need method='sum'")
        if not _all_concrete(weights, profile.rho_ye):
            raise ValueError("method='convolution' needs concrete weights and profile; use method='sum' "
                             "to differentiate with respect to them")
        (w, keep), rho_ye = _kept_rows(weights), np.asarray(profile.rho_ye, dtype=float)
        keep = keep & (rho_ye > 0)
        c = _convolution_mixing(u, msq, energy_GeV, rho_ye[keep], w[keep] / w[keep].sum(), alpha, anti, nsi_mat, na,
                                label)
    elif method == "sum":
        r = jnp.asarray(profile.r_over_rsun)
        if _is_concrete(weights):
            # Fixed by the weights alone, so the profile stays differentiable.
            w_np, keep = _kept_rows(weights)
            w, r = jnp.asarray(w_np[keep]), r[np.flatnonzero(keep)]
        else:
            w = jnp.asarray(weights)
            w = jnp.where(w > _WEIGHT_FLOOR * jnp.max(w), w, 0.0)
        v_cc, v_nc = _potentials_eV(profile, r)
        w = w / jnp.sum(w)
        v_max = jnp.max(jnp.abs(jnp.concatenate([v_cc, v_nc])))

        def at_energy(e_eV):
            shift = _block_shift(u, msq, label, e_eV, v_max, nsi_mat, anti)
            vecs = jnp.linalg.eigh(_hamiltonians(params, e_eV, v_cc, v_nc, anti, nsi_mat) + shift)[1]
            return w @ (jnp.abs(vecs[:, alpha, :]) ** 2)

        energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
        c = at_energy(energy_eV) if energy_eV.ndim == 0 else jax.vmap(at_energy)(energy_eV)
    else:
        raise ValueError(f"unknown method {method!r}; choose 'sum', 'convolution' or 'quadratic'")
    if check_adiabaticity:
        _warn_if_nonadiabatic(params, energy_GeV, profile, weights, anti, nsi)
    # k-th matter eigenstate in (block, eigenvalue) order -> k-th mass state in (block, m^2) order.
    return jnp.zeros_like(c).at[..., mass_order].set(c)


def _warn_if_nonadiabatic(params, energy_GeV, profile, weights, anti, nsi):
    """Warn when the adiabaticity drops below ``GAMMA_WARN`` (concrete inputs only)."""
    leaves = jax.tree_util.tree_leaves((params, energy_GeV, profile, weights, nsi))
    if not all(_is_concrete(x) for x in leaves):
        return
    w = np.asarray(weights, dtype=float)
    r_min = float(np.asarray(profile.r_over_rsun)[np.argmax(w > _WEIGHT_FLOOR * w.max())])
    e = np.unique(np.asarray(energy_GeV, dtype=float).ravel())
    e = e[np.unique(np.linspace(0, len(e) - 1, min(len(e), _GAMMA_ENERGIES)).round().astype(int))]
    gamma = np.asarray(adiabaticity(params, e, profile, r_min=r_min, anti=anti, nsi=nsi))
    if np.min(gamma) < GAMMA_WARN:
        warnings.warn(f"production_mixing: adiabaticity parameter down to {np.min(gamma):.3g} at "
                      f"{1e3 * e[np.argmin(gamma)]:.3g} MeV (below {GAMMA_WARN:g}); Landau-Zener jumps make the "
                      "adiabatic result unreliable; use method='quadratic'", stacklevel=3)


def _profile_potentials(profile):
    """Interpolator ``x -> (V_CC, dV_CC/dr, V_NC, dV_NC/dr)`` of the profile, and its radial range."""
    r = np.asarray(profile.r_over_rsun, dtype=float)
    v_cc = np.asarray(C.matter_potential_eV(np.asarray(profile.rho_ye, dtype=float), 1.0))
    v_nc = (np.zeros_like(r) if profile.rho_yn is None
            else 0.5 * np.asarray(C.matter_potential_eV(np.asarray(profile.rho_yn, dtype=float), 1.0)))
    mid = 0.5 * (r[1:] + r[:-1])
    slopes = [np.diff(v) / np.diff(r) for v in (v_cc, v_nc)]

    def pot(x):
        x = np.clip(x, r[0], r[-1])
        return [f for v, d in zip((v_cc, v_nc), slopes) for f in (np.interp(x, r, v), np.interp(x, mid, d))]
    return pot, float(r[0]), float(r[-1])


def _matter_terms(u, msq, anti, nsi, n_active):
    """``(m_vac, m_cc, m_nc)`` with ``H = m_vac / (2E) + V_CC m_cc + V_NC m_nc`` in the flavor basis."""
    def h(v, vn):
        return matter_hamiltonian(u, msq, 0.5, v, anti=anti, nsi=nsi, v_nc_eV=vn, n_active=n_active)

    m_vac = h(0.0, 0.0)
    return m_vac, h(1.0, 0.0) - m_vac, h(0.0, 1.0) - m_vac


# --- method="quadratic": interaction-picture Magnus steps on a fixed grid ------------------------------


def _path_mixing(params, energy_GeV, profile, weights, alpha, anti, nsi, n_active, method):
    """``production_mixing`` without the adiabatic assumption (see its docstring)."""
    if not _all_concrete(weights, profile):
        raise ValueError(f"method={method!r} needs a concrete profile and weights (they fix its radial grid)")
    u, msq = params.pmns(), params.msquared()
    w, keep = _kept_rows(weights)
    r_prod = np.asarray(profile.r_over_rsun, dtype=float)[keep]
    pot, lo, hi = _profile_potentials(profile)
    edges = _quadratic_grid(pot, lo, hi, r_prod, msq.shape[0], n_active, profile.R_sun_km * C.KM_TO_INV_EV)
    nodes = np.empty(2 * len(edges) - 1)
    nodes[0::2], nodes[1::2] = edges, 0.5 * (edges[1:] + edges[:-1])
    v_cc, _, v_nc, _ = pot(nodes)
    k_prod = np.searchsorted(edges, r_prod)
    assert np.array_equal(edges[k_prod], r_prod)                     # production radii are edges
    w_edge = np.bincount(k_prod, weights=w[keep], minlength=len(edges))
    return _path_fractions(u, msq, jnp.asarray(v_cc), jnp.asarray(v_nc),
                             jnp.asarray(np.diff(edges) * profile.R_sun_km * C.KM_TO_INV_EV),
                             jnp.asarray(w_edge / w_edge.sum()), nsi, jnp.asarray(energy_GeV) * C.GEV_TO_EV,
                             alpha=alpha, anti=anti, n_active=n_active)


def _quadratic_grid(pot, lo, hi, r_prod, n, n_active, scale):
    """Step edges with density ``sqrt(||dH/dr|| / eps)`` and every production radius as an edge."""
    # CC and NC terms only (not NSI), in numpy, so the grid stays concrete under jit.
    m_cc = np.zeros((n, n))
    m_cc[0, 0] = 1.0
    m_nc = np.diag((np.arange(n) >= n_active).astype(float))
    r = np.linspace(r_prod.min(), hi, _QUADRATIC_GRID_SAMPLES)
    _, dv_cc, _, dv_nc = pot(r)
    dh = (dv_cc[:, None, None] * m_cc + dv_nc[:, None, None] * m_nc) * scale
    dh = dh - np.einsum("rii->r", dh)[:, None, None] * np.eye(n) / n
    dens = np.sqrt(np.sqrt(np.sum(np.abs(dh) ** 2, axis=(1, 2))) / _QUADRATIC_EPS)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(r))])
    m = int(max(_QUADRATIC_MIN_STEPS, np.ceil(cdf[-1])))
    return _with_production_edges(np.interp(np.linspace(0.0, cdf[-1], m + 1), cdf, r), r_prod, lo, hi)


def _with_production_edges(q, r_prod, lo, hi):
    """Edges ``q`` plus every production radius and ``hi``, dropping edges within 1e-9 of those."""
    fixed = np.unique(np.append(r_prod, hi))
    i = np.searchsorted(fixed, q).clip(1, len(fixed) - 1)
    far = np.minimum(np.abs(q - fixed[i - 1]), np.abs(q - fixed[i])) > 1e-9 * (hi - lo)
    return np.union1d(q[far], fixed)


@functools.partial(jax.jit, static_argnames=("alpha", "anti", "n_active"))
def _path_fractions(u, msq, v_cc, v_nc, length, w_edge, nsi, energy_eV, alpha, anti, n_active):
    """Mass-state fractions averaged over the edge weights, shape ``([n_E,] N)``."""
    # One energy at a time, recomputed in the backward pass: reverse-mode memory independent of n_E.
    one = jax.checkpoint(functools.partial(_path_one, u, msq, v_cc, v_nc, length, w_edge, nsi, alpha=alpha,
                                           anti=anti, n_active=n_active))
    return one(energy_eV) if energy_eV.ndim == 0 else jax.lax.map(one, energy_eV)


def _path_one(u, msq, v_cc, v_nc, length, w_edge, nsi, energy_eV, alpha, anti, n_active):
    m_vac, m_cc, m_nc = _matter_terms(u, msq, anti, nsi, n_active)
    h = m_vac / (2.0 * energy_eV) + v_cc[:, None, None] * m_cc + v_nc[:, None, None] * m_nc
    hl, hm, hr = h[0:-1:2], h[1::2], h[2::2]
    t = length[:, None, None]
    w, q = _eigh_safe(hm)
    qh = _dag(q)
    h1 = qh @ ((hr - hl) / t) @ q                                    # delta(s) = h1 s + h2 s^2 / 2 about the midpoint
    h2 = qh @ (4.0 * (hr - 2.0 * hm + hl) / t ** 2) @ q
    m1, m2 = _ip_moments((w[:, :, None] - w[:, None, :]) * t, t)
    half = _expmh(hm * (0.5 * t))
    steps = half @ (q @ _expmh(h1 * m1 + 0.5 * h2 * m2) @ qh) @ half
    tails = jax.lax.associative_scan(lambda a, b: a @ b, steps, reverse=True)   # S(surface <- edge k)
    tails = jnp.concatenate([tails, jnp.eye(tails.shape[-1], dtype=tails.dtype)[None]])
    udag = (u if anti else jnp.conj(u)).T
    amp = jnp.einsum("ia,kab->kib", udag, tails)[:, :, alpha]
    return w_edge @ (jnp.abs(amp) ** 2)


def _ip_moments(t, length):
    """``M_p = int_{-h/2}^{h/2} s^p e^{i theta s} ds`` (p = 1, 2) for ``t = theta h``, via
    ``E_k(t) = int_0^1 u^k e^{itu} du``: power series for ``|t| < 1``, recursion above."""
    small = jnp.abs(t) < 1.0
    ts, tz = jnp.where(small, 1.0, t), jnp.where(small, t, 0.0)
    fact = np.cumprod([1.0] + list(range(1, 20)))
    e, ek = [], (jnp.exp(1j * ts) - 1.0) / (1j * ts)
    for k in range(3):
        if k:
            ek = jnp.exp(1j * ts) / (1j * ts) - k / (1j * ts) * ek
        e.append(jnp.where(small, sum((1j * tz) ** p / (fact[p] * (k + p + 1)) for p in range(20)), ek))
    ph = jnp.exp(-0.5j * t)
    return length ** 2 * ph * (e[1] - 0.5 * e[0]), length ** 3 * ph * (e[2] - e[1] + 0.25 * e[0])


@jax.custom_jvp
def _eigh_safe(a):
    """``eigh`` whose derivative drops exactly degenerate pairs (levels of uncoupled blocks); only functions of
    the matrix are used, so those terms cancel."""
    return tuple(jnp.linalg.eigh(a))


@_eigh_safe.defjvp
def _eigh_safe_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    w, v = jnp.linalg.eigh(a)
    vh = _dag(v)
    x = vh @ da @ v
    gap = w[..., None, :] - w[..., :, None]
    keep = jnp.abs(gap) > 1e-13 * jnp.max(jnp.abs(w), axis=-1)[..., None, None]   # relative: H is in eV
    f = jnp.where(keep, 1.0 / (gap + (~keep).astype(gap.dtype)), 0.0)
    return (w, v), (jnp.real(jnp.diagonal(x, axis1=-2, axis2=-1)), v @ (f * x))


@jax.custom_jvp
def _expmh(x):
    """``exp(-i X)`` for Hermitian ``X``, by eigendecomposition."""
    w, v = jnp.linalg.eigh(x)
    return (v * jnp.exp(-1j * w)[..., None, :]) @ _dag(v)


@_expmh.defjvp
def _expmh_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    w, v = jnp.linalg.eigh(x)
    vh = _dag(v)
    a, b = w[..., :, None], w[..., None, :]
    dd = -1j * jnp.exp(-0.5j * (a + b)) * jnp.sinc((a - b) / (2.0 * np.pi))   # (e^-ia - e^-ib)/(a - b), stable
    return (v * jnp.exp(-1j * w)[..., None, :]) @ vh, v @ (dd * (vh @ dx @ v)) @ vh


def probability_solar(params, energy_GeV, profile, weights, cos_zenith=None, alpha=0, anti=False, nsi=None,
                      flavor_out=None, method="sum", check_adiabaticity=True, **earth_options):
    """``P(nu_alpha -> nu_beta)`` for solar neutrinos from one source, by day or at night.

    Arguments as in :func:`production_mixing`. With ``cos_zenith`` (the Sun's, negative
    at night) the mass states cross the Earth as in :func:`probability_earth`, which
    takes ``earth_options``. Returns ``P[..., beta]``, shape ``([n_E,] N)`` by day and
    ``([n_cz,] [n_E,] N)`` at night, or the ``flavor_out`` entry.
    """
    _check_flavor_index(flavor_out, n_flavors(params), "flavor_out")
    if cos_zenith is None and earth_options:
        raise TypeError(f"Earth options {sorted(earth_options)} need cos_zenith; by day there is no Earth crossing")
    c = production_mixing(params, energy_GeV, profile, weights, alpha=alpha, anti=anti, nsi=nsi, method=method,
                          check_adiabaticity=check_adiabaticity)
    if cos_zenith is None:
        p = c @ (jnp.abs(params.pmns()) ** 2).T
    else:
        p_mass = probability_earth(params, energy_GeV, cos_zenith, anti=anti, nsi=nsi, mass_states=True,
                                   **earth_options)
        p = jnp.sum(p_mass * c[..., None, :], axis=-1)
    return p if flavor_out is None else p[..., flavor_out]


def _convolution_mixing(u, msq, energy_GeV, rho_ye, w, alpha, anti, nsi, n_active, label):
    """``production_mixing`` by matter eigenstate, as a correlation in ln A."""
    energy = np.asarray(energy_GeV, dtype=float)
    if not np.all(np.isfinite(energy)):
        raise ValueError("method='convolution' needs concrete, finite energies")
    # A = 2 E V_CC = k E[GeV] rho_ye, so ln A = ln(k E) + ln rho_ye.
    k = 2.0 * C.GEV_TO_EV * C.matter_potential_eV(1.0, 1.0)
    v = np.log(rho_ye)
    v_min = v.min()
    m = int(np.ceil((v.max() - v_min) / _DU)) + 2
    pos = (v - v_min) / _DU
    lo = np.floor(pos).astype(int)
    kernel = np.zeros(m)
    np.add.at(kernel, lo, w * (1.0 - (pos - lo)))
    np.add.at(kernel, lo + 1, w * (pos - lo))
    u_e = np.log(k * energy.ravel())
    u_lo = np.floor((u_e.min() - 0.01) / _DU) * _DU
    n_out = int(np.ceil((u_e.max() + 0.01 - u_lo) / _DU)) + 1
    n_pad = n_out + m - 1
    A = jnp.asarray(np.exp(u_lo + v_min + _DU * np.arange(n_pad)))
    # Eigenvectors of H and of 2 E H = U diag(msq) U^dag + A (matter matrix) coincide: E = 1/2 eV, V = A.
    shift = _block_shift(u, msq, label, 0.5, A[-1], nsi, anti)
    h = jax.vmap(lambda a: jnp.abs(_matter_eigvecs(u, msq, 0.5, a, 0.0, anti, nsi, n_active, shift)[alpha]) ** 2)(A)
    n_fft = int(2 ** np.ceil(np.log2(n_pad + m - 1)))
    kernel_fft = jnp.asarray(np.fft.rfft(kernel[::-1], n_fft))
    corr = jnp.fft.irfft(jnp.fft.rfft(h.T, n_fft) * kernel_fft, n_fft)[:, m - 1:n_pad]
    grid = jnp.asarray(u_lo + _DU * np.arange(n_out))
    out = jax.vmap(jnp.interp, in_axes=(None, None, 0))(jnp.asarray(u_e), grid, corr).T
    return out.reshape(energy.shape + (corr.shape[0],))
