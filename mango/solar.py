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

from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

from . import constants as C
from .hamiltonian import matter_hamiltonian
from .oscillator import _check_flavor_index, _n_active, _nsi_matrix, n_flavors, probability_earth

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


def production_mixing(params, energy_GeV, profile, weights, alpha=0, anti=False, nsi=None, method="sum"):
    """Adiabatic vacuum mass-state fractions of solar neutrinos, averaged over a production region.

    ``weights`` are one source's production weights on ``profile.r_over_rsun``
    (any normalisation). Returns ``c[..., i]`` with shape ``([n_E,] N)``: the
    fraction of neutrinos produced as flavor ``alpha`` that leave the Sun as vacuum
    mass eigenstate ``i``. Matter eigenstates are followed out of the Sun without
    level crossings, so the k-th lowest at production leaves as the k-th lightest
    mass state; some sterile or NSI textures cross levels and break this. By day
    ``P(nu_alpha -> nu_beta) = sum_i c_i |U_{beta i}|^2``; at night use
    ``probability_earth(..., mass_states=True)`` in place of ``|U|^2``.

    ``method="sum"`` diagonalises the Hamiltonian at every radius with weight above
    1e-14 of the peak. ``method="convolution"`` uses that the matter term is
    ``n_e`` times a fixed matrix (also with NSI and for antineutrinos, but not for
    sterile flavors with ``rho_yn``): the composition then depends on energy and
    density only through ``A = 2 E V_CC``, and the average is a correlation in
    ``ln A``, computed by FFT on a grid of spacing 5e-4 (error ~1e-8). It needs
    concrete energies; gradients in ``params`` are exact.
    """
    w = np.asarray(weights, dtype=float)
    rho_ye = np.asarray(profile.rho_ye, dtype=float)
    keep = (w > _WEIGHT_FLOOR * w.max()) & (rho_ye > 0)
    w = w[keep] / w[keep].sum()
    u, msq = params.pmns(), params.msquared()
    n, na = msq.shape[0], _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)

    if method == "convolution":
        if profile.rho_yn is not None and na < n:
            raise ValueError("method='convolution' needs a matter term proportional to n_e; "
                             "sterile flavors with rho_yn need method='sum'")
        c = _convolution_mixing(u, msq, energy_GeV, rho_ye[keep], w, alpha, anti, nsi_mat, na)
    elif method == "sum":
        v_cc = C.matter_potential_eV(jnp.asarray(rho_ye[keep]), 1.0)
        v_nc = (jnp.zeros_like(v_cc) if profile.rho_yn is None
                else 0.5 * C.matter_potential_eV(jnp.asarray(np.asarray(profile.rho_yn)[keep]), 1.0))

        def at_energy(e_eV):
            vecs = jax.vmap(lambda v, vn: _matter_eigvecs(u, msq, e_eV, v, vn, anti, nsi_mat, na))(v_cc, v_nc)
            return jnp.asarray(w) @ (jnp.abs(vecs[:, alpha, :]) ** 2)

        energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
        c = at_energy(energy_eV) if energy_eV.ndim == 0 else jax.vmap(at_energy)(energy_eV)
    else:
        raise ValueError(f"unknown method {method!r}; choose 'sum' or 'convolution'")
    # k-th lowest matter eigenstate -> k-th lightest mass state.
    return jnp.zeros_like(c).at[..., jnp.argsort(msq)].set(c)



def probability_solar(params, energy_GeV, profile, weights, cos_zenith=None, anti=False, nsi=None,
                      flavor_out=None, method="sum", **earth_options):
    """``P(nu_e -> nu_beta)`` for solar neutrinos from one source, by day or at night.

    ``profile`` and ``weights`` are as in :func:`production_mixing`, and so is
    ``method``. Neutrinos leave the Sun as an incoherent mixture of vacuum mass
    states; by day (``cos_zenith=None``) they reach the detector in vacuum, at
    night they cross the Earth as in :func:`probability_earth`, which takes the
    remaining keyword arguments (``det_depth_km``, ``earth_model``, ``n_sub``, ...).
    Returns ``P[..., beta]`` with shape ``([n_E,] N)`` by day and
    ``([n_cz,] [n_E,] N)`` at night, or the ``flavor_out`` entry alone.
    """
    _check_flavor_index(flavor_out, n_flavors(params), "flavor_out")
    c = production_mixing(params, energy_GeV, profile, weights, anti=anti, nsi=nsi, method=method)
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
    out = jnp.stack([jnp.interp(jnp.asarray(u_e), grid, corr[c]) for c in range(corr.shape[0])], axis=-1)
    return out.reshape(energy.shape + (corr.shape[0],))
