"""Solar matter profile and adiabatic MSW evolution.

Provides a standard-solar-model electron-density profile (loaded from a BS05-style
table, or an analytic exponential approximation) and the **adiabatic** mass-state
composition of a neutrino produced inside the Sun -- the mechanism behind the LMA
solar solution. Idea adapted from the nu-waves library. :func:`production_mixing`
averages that composition over a source's production region, and
:func:`probability_solar` turns it into probabilities by day and at night.

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
from .layers import propagate_layers
from .oscillator import (_check_flavor_index, _is_concrete, _n_active, _nsi_matrix, _resolve_backend, n_flavors,
                         probability_earth)

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


def _matter_eigvecs(u, msq, energy_eV, v_cc, v_nc, anti, nsi, n_active):
    """Matter eigenstates in the flavor basis (columns, ascending eigenvalue)."""
    h = matter_hamiltonian(u, msq, energy_eV, v_cc, anti=anti, nsi=nsi, v_nc_eV=v_nc, n_active=n_active)
    return jnp.linalg.eigh(h)[1]


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

    def eigvecs(r):
        # matter_hamiltonian applies U -> U* and V -> -V itself for antineutrinos.
        v_cc, v_nc = _potentials_eV(profile, jnp.asarray(r) / profile.R_sun_km)
        return _matter_eigvecs(u, msq, energy_eV, v_cc, v_nc, anti, None, na)

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


def _hamiltonians(params, energy_eV, v_cc, v_nc, anti, nsi):
    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    return jax.vmap(lambda v, vn: matter_hamiltonian(u, msq, energy_eV, v, anti=anti, nsi=nsi, v_nc_eV=vn,
                                                     n_active=na))(v_cc, v_nc)


def adiabaticity(params, energy_GeV, profile, r_min=0.0, r_max=np.inf, anti=False, nsi=None):
    """Smallest adiabaticity parameter between ``r_min`` and ``r_max`` [R_sun], shape ``([n_E])``.

    ``gamma_ij = (lambda_i - lambda_j)^2 / (2 |<i|dH/dx|j>|)``, minimised over radii
    and pairs of matter eigenstates; at a two-flavor resonance this is the usual
    ``Delta m^2 sin^2(2 theta) / (2 E cos(2 theta) |d ln n_e / dx|)``. The adiabatic
    approximation is good to ~1e-5 above gamma ~ 10 and fails at gamma ~ 1-3.
    """
    r = np.asarray(profile.r_over_rsun, dtype=float)
    sel = np.flatnonzero((r >= r_min) & (r <= r_max))
    x = jnp.asarray(r[sel] * profile.R_sun_km * C.KM_TO_INV_EV)
    v_cc, v_nc = _potentials_eV(profile, jnp.asarray(r[sel]))
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    return _gamma_min(params, jnp.atleast_1d(energy_eV), x, v_cc, v_nc, _nsi_matrix(nsi, _n_active(params)),
                      anti).reshape(energy_eV.shape)


@functools.partial(jax.jit, static_argnames="anti")
def _gamma_min(params, energy_eV, x, v_cc, v_nc, nsi, anti):
    """Minimum over radii and pairs of matter eigenstates of the adiabaticity parameter, per energy."""
    iu = np.triu_indices(params.msquared().shape[0], 1)

    def at_energy(e_eV):
        h = _hamiltonians(params, e_eV, v_cc, v_nc, anti, nsi)
        lam, vec = jnp.linalg.eigh(0.5 * (h[1:] + h[:-1]))
        dh = (h[1:] - h[:-1]) / jnp.diff(x)[:, None, None]
        coupling = jnp.abs(jnp.einsum("rai,rab,rbj->rij", jnp.conj(vec), dh, vec))[:, iu[0], iu[1]]
        gap2 = ((lam[:, :, None] - lam[:, None, :]) ** 2)[:, iu[0], iu[1]]
        return jnp.min(jnp.where(coupling > 0, gap2 / (2.0 * jnp.where(coupling > 0, coupling, 1.0)), jnp.inf))

    return jax.vmap(at_energy)(energy_eV)


def production_mixing(params, energy_GeV, profile, weights, alpha=0, anti=False, nsi=None, method="sum",
                      outer="adiabatic", n_layers=5000):
    """Vacuum mass-state fractions of solar neutrinos, averaged over a production region.

    ``weights`` are production fractions per point of ``profile.r_over_rsun`` (already
    integrated over each grid shell, not a density dPhi/dr). Returns ``c[..., i]``,
    shape ``([n_E,] N)``: the fraction produced as ``alpha`` that leaves the Sun as
    mass state ``i``. By day ``P(nu_alpha -> nu_beta) = sum_i c_i |U_{beta i}|^2``; at
    night see :func:`probability_solar`.

    Neutrinos are assumed to follow their matter eigenstates (non-crossing, so the
    k-th lowest leaves as the k-th lightest mass state). This fails at narrow avoided
    crossings, e.g. small active-sterile mixing at Delta m^2_41 ~ 1e-5-1e-4 eV^2; on
    concrete inputs the function warns when :func:`adiabaticity` drops below
    ``GAMMA_WARN``. With ``outer="exact"`` (Parke's factorisation, PRL 57, 1275
    (1986)) only the production region is treated adiabatically: from its outer edge
    the matter states are propagated exactly through ``n_layers`` layers, and combined
    incoherently. This handles resonances outside the production region, as for those
    steriles; it needs concrete weights.

    ``method="sum"`` diagonalises at every radius and is differentiable in ``params``,
    the profile and the weights. ``method="convolution"`` needs a matter term equal to
    ``n_e`` times a fixed matrix (NSI defined relative to ``n_e`` and antineutrinos
    included; not steriles with ``rho_yn``): the composition then depends only on
    ``A = 2 E V_CC``, and the average is an FFT correlation in ``ln A`` (error ~1e-8).
    It needs concrete energies, profile and weights; gradients in ``params`` are exact.
    """
    _check_weights(weights, profile)
    u, msq = params.pmns(), params.msquared()
    n, na = msq.shape[0], _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)

    if method == "convolution":
        if profile.rho_yn is not None and na < n:
            raise ValueError("method='convolution' needs a matter term proportional to n_e; "
                             "sterile flavors with rho_yn need method='sum'")
        if not (_is_concrete(weights) and _is_concrete(profile.rho_ye)):
            raise ValueError("method='convolution' needs concrete weights and profile; use method='sum' "
                             "to differentiate with respect to them")
        w, rho_ye = np.asarray(weights, dtype=float), np.asarray(profile.rho_ye, dtype=float)
        keep = (w > _WEIGHT_FLOOR * w.max()) & (rho_ye > 0)
        c = _convolution_mixing(u, msq, energy_GeV, rho_ye[keep], w[keep] / w[keep].sum(), alpha, anti, nsi_mat, na)
    elif method == "sum":
        r = jnp.asarray(profile.r_over_rsun)
        if _is_concrete(weights):
            # Fixed by the weights alone, so the profile stays differentiable.
            w_np = np.asarray(weights, dtype=float)
            keep = np.flatnonzero(w_np > _WEIGHT_FLOOR * w_np.max())
            w, r = jnp.asarray(w_np[keep]), r[keep]
        else:
            w = jnp.asarray(weights)
            w = jnp.where(w > _WEIGHT_FLOOR * jnp.max(w), w, 0.0)
        v_cc, v_nc = _potentials_eV(profile, r)
        w = w / jnp.sum(w)

        def at_energy(e_eV):
            vecs = jnp.linalg.eigh(_hamiltonians(params, e_eV, v_cc, v_nc, anti, nsi_mat))[1]
            return w @ (jnp.abs(vecs[:, alpha, :]) ** 2)

        energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
        c = at_energy(energy_eV) if energy_eV.ndim == 0 else jax.vmap(at_energy)(energy_eV)
    else:
        raise ValueError(f"unknown method {method!r}; choose 'sum' or 'convolution'")
    if outer == "exact":
        r_split = _production_edge(weights, profile)
        _warn_if_nonadiabatic(params, energy_GeV, profile, weights, anti, nsi, r_max=r_split)
        t = _outer_transitions(params, energy_GeV, profile, r_split, anti, nsi_mat, n_layers)
        return jnp.einsum("...ik,...k->...i", t, c)
    if outer != "adiabatic":
        raise ValueError(f"unknown outer {outer!r}; choose 'adiabatic' or 'exact'")
    _warn_if_nonadiabatic(params, energy_GeV, profile, weights, anti, nsi)
    # k-th lowest matter eigenstate -> k-th lightest mass state.
    return jnp.zeros_like(c).at[..., jnp.argsort(msq)].set(c)


def _production_edge(weights, profile, tail=1e-6):
    """Radius [R_sun] inside which all but ``tail`` of the production lies."""
    if not _is_concrete(weights):
        raise ValueError("outer='exact' needs concrete weights")
    w = np.asarray(weights, dtype=float)
    return float(np.asarray(profile.r_over_rsun)[np.searchsorted(np.cumsum(w) / w.sum(), 1.0 - tail)])


def _outer_transitions(params, energy_GeV, profile, r_split, anti, nsi, n_layers):
    """``|<nu_i| S |nu_k^m(r_split)>|^2`` from ``r_split`` to the surface, shape ``([n_E,] N, N)``."""
    edges = jnp.linspace(r_split, float(np.asarray(profile.r_over_rsun)[-1]), n_layers + 1)
    v_cc, v_nc = _potentials_eV(profile, 0.5 * (edges[1:] + edges[:-1]))
    v0_cc, v0_nc = _potentials_eV(profile, jnp.asarray(r_split))
    length = jnp.diff(edges) * profile.R_sun_km * C.KM_TO_INV_EV
    u, msq, na = params.pmns(), params.msquared(), _n_active(params)
    u_in = jnp.conj(u) if anti else u
    backend = _resolve_backend(params, "cayley")

    def at_energy(e_eV):
        s = propagate_layers(u, msq, e_eV, v_cc, length, anti=anti, backend=backend, nsi=nsi, v_nc=v_nc, n_active=na)
        vecs = _matter_eigvecs(u, msq, e_eV, v0_cc, v0_nc, anti, nsi, na)
        return jnp.abs(jnp.conj(u_in).T @ s @ vecs) ** 2

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    return at_energy(energy_eV) if energy_eV.ndim == 0 else jax.vmap(at_energy)(energy_eV)


def _warn_if_nonadiabatic(params, energy_GeV, profile, weights, anti, nsi, r_max=np.inf):
    """Warn when the adiabaticity drops below ``GAMMA_WARN`` (concrete inputs only)."""
    leaves = jax.tree_util.tree_leaves((params, energy_GeV, profile, weights, nsi))
    if not all(_is_concrete(x) for x in leaves):
        return
    w = np.asarray(weights, dtype=float)
    r_min = float(np.asarray(profile.r_over_rsun)[np.argmax(w > _WEIGHT_FLOOR * w.max())])
    e = np.unique(np.asarray(energy_GeV, dtype=float).ravel())
    e = e[np.unique(np.linspace(0, len(e) - 1, min(len(e), _GAMMA_ENERGIES)).round().astype(int))]
    gamma = np.asarray(adiabaticity(params, e, profile, r_min=r_min, r_max=r_max, anti=anti, nsi=nsi))
    if np.min(gamma) < GAMMA_WARN:
        warnings.warn(f"production_mixing: adiabaticity parameter down to {np.min(gamma):.3g} at "
                      f"{1e3 * e[np.argmin(gamma)]:.3g} MeV (below {GAMMA_WARN:g}); Landau-Zener jumps make the "
                      "adiabatic result unreliable" + (" inside the production region" if np.isfinite(r_max) else
                                                       "; try outer='exact'"), stacklevel=3)


def probability_solar(params, energy_GeV, profile, weights, cos_zenith=None, alpha=0, anti=False, nsi=None,
                      flavor_out=None, method="sum", outer="adiabatic", **earth_options):
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
                          outer=outer)
    if cos_zenith is None:
        p = c @ (jnp.abs(params.pmns()) ** 2).T
    else:
        p_mass = probability_earth(params, energy_GeV, cos_zenith, anti=anti, nsi=nsi, mass_states=True,
                                   **earth_options)
        p = jnp.sum(p_mass * c[..., None, :], axis=-1)
    return p if flavor_out is None else p[..., flavor_out]


def _convolution_mixing(u, msq, energy_GeV, rho_ye, w, alpha, anti, nsi, n_active):
    """``production_mixing`` by ascending matter eigenstate, as a correlation in ln A."""
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
    h = jax.vmap(lambda a: jnp.abs(_matter_eigvecs(u, msq, 0.5, a, 0.0, anti, nsi, n_active)[alpha]) ** 2)(A)
    n_fft = int(2 ** np.ceil(np.log2(n_pad + m - 1)))
    kernel_fft = jnp.asarray(np.fft.rfft(kernel[::-1], n_fft))
    corr = jnp.fft.irfft(jnp.fft.rfft(h.T, n_fft) * kernel_fft, n_fft)[:, m - 1:n_pad]
    grid = jnp.asarray(u_lo + _DU * np.arange(n_out))
    out = jax.vmap(jnp.interp, in_axes=(None, None, 0))(jnp.asarray(u_e), grid, corr).T
    return out.reshape(energy.shape + (corr.shape[0],))
