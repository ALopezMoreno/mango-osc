"""Solar matter profile and adiabatic MSW evolution.

Provides a standard-solar-model electron-density profile (loaded from a BS05-style
table, or an analytic exponential approximation) and the **adiabatic** mass-state
composition of a neutrino produced inside the Sun -- the mechanism behind the LMA
solar solution. Idea adapted from the nu-waves library.

For predictions, :func:`load_production` gives bundled production distributions
per source with the electron and neutron densities of two standard solar models,
:func:`production_mixing` averages the adiabatic composition over a source's
production region, and :func:`solar_probabilities` combines it with propagation
through the Earth at night.

In the adiabatic regime a neutrino produced as flavor ``alpha`` populates the
instantaneous matter eigenstates with fixed weights ``w_k = |<nu_k^m|nu_alpha>|^2``
(no level hopping); the observable vacuum mass-state fractions then evolve only
because the matter eigenstates rotate as the density drops:

    F_i(r) = sum_k w_k |<nu_i^vac | nu_k^m(r)>|^2 .
"""

from __future__ import annotations

import os
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

from . import constants as C
from .hamiltonian import matter_hamiltonian

R_SUN_KM = 695700.0

# ``rho_yn`` is the neutron density rho (1 - Y_e) in the units of ``rho_ye``, or None.
SolarProfile = namedtuple("SolarProfile", ["r_over_rsun", "rho_ye", "R_sun_km", "rho_yn"],
                          defaults=(None,))

SOURCES = ("pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F")
SOLAR_MODELS = ("SF3_GS98", "BS05_OP")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "solar")

# Radial production distributions and densities [N_A/cm^3]; ``weights`` maps each
# source to its production weight per radius, summing to one.
SolarProduction = namedtuple("SolarProduction", ["r_over_rsun", "n_e", "n_n", "weights"])


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

    u = params.pmns()
    if anti:
        u = jnp.conj(u)
    msq = params.msquared()
    energy_eV = energy_GeV * C.GEV_TO_EV

    n_active = getattr(params, "n_active", msq.shape[0])

    def eigvecs(r):
        v_eV = potential_eV(profile, r)
        v_nc = 0.0 if profile.rho_yn is None else 0.5 * C.matter_potential_eV(jnp.interp(
            jnp.asarray(r) / profile.R_sun_km, jnp.asarray(profile.r_over_rsun), jnp.asarray(profile.rho_yn)), 1.0)
        h = matter_hamiltonian(u, msq, energy_eV, v_eV, anti=anti, v_nc_eV=v_nc, n_active=n_active)
        _w, vecs = jnp.linalg.eigh(h)
        return vecs  # columns = matter eigenstates (flavor basis), asc. eigenvalue

    # Production weights w_k = |<nu_k^m(r_emit) | nu_alpha>|^2.
    vecs_emit = eigvecs(r_emit_km)
    w = jnp.abs(vecs_emit[alpha, :]) ** 2  # (N,)

    udag = jnp.conj(u).T

    def frac_at(r):
        vecs = eigvecs(r)
        a = udag @ vecs  # <nu_i^vac | nu_k^m> = (U^dag V)[i,k]
        return (jnp.abs(a) ** 2) @ w  # F_i = sum_k |a_ik|^2 w_k

    fractions = jax.vmap(frac_at)(r_km)
    return fractions[0] if scalar_r else fractions


# --- production-averaged predictions -----------------------------------------

def load_production(model="SF3_GS98"):
    """Bundled production distributions and densities of a standard solar model.

    ``model`` is "SF3_GS98" (Herrera & Serenelli, SF-III) or "BS05_OP" (Bahcall,
    Serenelli & Basu 2005). Returns a :class:`SolarProduction`; the provenance is
    in the header of ``mango/data/solar/production_<model>.dat``.
    """
    if model not in SOLAR_MODELS:
        raise ValueError(f"unknown solar model {model!r}; choose from {SOLAR_MODELS}")
    t = np.loadtxt(os.path.join(_DATA_DIR, f"production_{model}.dat"))
    return SolarProduction(r_over_rsun=t[:, 0], n_e=t[:, 1], n_n=t[:, 2],
                           weights={s: t[:, 3 + i] / t[:, 3 + i].sum() for i, s in enumerate(SOURCES)})


_WEIGHT_FLOOR = 1e-14   # production weight, relative to the maximum, below which radii are dropped


def _production_grid(production, source):
    w = np.asarray(production.weights[source], dtype=float)
    keep = (w > _WEIGHT_FLOOR * w.max()) & (np.asarray(production.n_e) > 0)
    return (np.asarray(production.n_e)[keep], np.asarray(production.n_n)[keep], w[keep] / w[keep].sum())


def production_mixing(params, energy_GeV, production, source="8B", alpha=0, anti=False,
                      nsi=None, method="sum"):
    """Adiabatic composition of solar neutrinos, averaged over a source's production region.

    Returns ``c[..., k] = sum_r f(r) |<nu_k^m(r)|nu_alpha>|^2`` with shape
    ``([n_E,] N)``: the fraction of neutrinos produced as flavor ``alpha`` in matter
    eigenstate ``k`` (ascending eigenvalue), weighted by the production weights
    ``f`` of ``source``. Adiabatically, matter eigenstate ``k`` leaves the Sun as
    vacuum mass eigenstate ``k``; this assumes no level crossing on the way out,
    which can fail for some sterile or NSI textures. Sterile flavors feel the
    neutral-current term from ``production.n_n``.

    ``method="sum"`` sums over the tabulated radii. ``method="convolution"`` is a
    fast path for three flavors with only the charged-current potential (no NSI,
    no antineutrinos): there the composition depends on energy and density only
    through ``A = 2 E V_CC``, and the average is a correlation in ``ln A``,
    computed by FFT on a grid of spacing 5e-4 (agrees with the sum to about
    1e-8). It needs concrete energies.
    """
    n_e, n_n, w = _production_grid(production, source)
    u, msq = params.pmns(), params.msquared()
    n = msq.shape[0]
    n_active = getattr(params, "n_active", n)
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    if method == "convolution":
        if n != 3 or nsi is not None or anti:
            raise ValueError("method='convolution' needs three flavors, no NSI and neutrinos; use method='sum'")
        return _convolution_mixing(u, msq, energy_GeV, n_e, w, alpha)
    if method != "sum":
        raise ValueError(f"unknown method {method!r}; choose 'sum' or 'convolution'")
    nsi_mat = None if nsi is None else (nsi.matrix(n_active) if hasattr(nsi, "matrix")
                                       else jnp.asarray(nsi, dtype=jnp.complex128))
    v_cc = C.matter_potential_eV(jnp.asarray(n_e), 1.0)
    v_nc = 0.5 * C.matter_potential_eV(jnp.asarray(n_n), 1.0)

    def at_energy(e_eV):
        def at_radius(v, vn):
            h = matter_hamiltonian(u, msq, e_eV, v, anti=anti, nsi=nsi_mat, v_nc_eV=vn, n_active=n_active)
            return jnp.abs(jnp.linalg.eigh(h)[1][alpha]) ** 2
        return jnp.asarray(w) @ jax.vmap(at_radius)(v_cc, v_nc)

    return at_energy(energy_eV) if energy_eV.ndim == 0 else jax.vmap(at_energy)(energy_eV)


_DU = 5e-4   # ln(A) grid spacing of the convolution; interpolation error ~ _DU^2


def _convolution_mixing(u, msq, energy_GeV, n_e, w, alpha):
    """``production_mixing`` for three flavors with the charged-current potential only."""
    energy = np.asarray(energy_GeV, dtype=float)
    if not np.all(np.isfinite(energy)):
        raise ValueError("method='convolution' needs concrete, finite energies")
    # A = 2 E V_CC = k * E[GeV] * n_e, so ln A = ln(k E) + ln n_e.
    k = 2.0 * C.GEV_TO_EV * C.matter_potential_eV(1.0, 1.0)
    v = np.log(n_e)
    v_min = v.min()
    m = int(np.ceil((v.max() - v_min) / _DU)) + 1
    pos = (v - v_min) / _DU
    lo = np.clip(np.floor(pos).astype(int), 0, m - 2)
    kernel = np.zeros(m)
    np.add.at(kernel, lo, w * (1.0 - (pos - lo)))
    np.add.at(kernel, lo + 1, w * (pos - lo))
    u_e = np.log(k * energy.ravel())
    u_lo = np.floor((u_e.min() - 0.01) / _DU) * _DU
    n_out = int(np.ceil((u_e.max() + 0.01 - u_lo) / _DU)) + 1
    n_pad = n_out + m - 1
    A = jnp.asarray(np.exp(u_lo + v_min + _DU * np.arange(n_pad)))
    n_fft = int(2 ** np.ceil(np.log2(n_pad + m - 1)))
    kernel_fft = jnp.asarray(np.fft.rfft(kernel[::-1], n_fft))

    # Eigenvalues of H0 + A diag(1, 0, 0) from its characteristic cubic, linear in A; then
    # |<alpha|k>|^2 = C_alpha(l_k) / prod_{j != k} (l_k - l_j), C_alpha the alpha-alpha cofactor of l - H.
    h0 = (u * msq) @ jnp.conj(u).T
    d = jnp.real(jnp.diag(h0))
    s01, s02, s12 = (jnp.abs(h0[i, j]) ** 2 for i, j in ((0, 1), (0, 2), (1, 2)))
    c_ee = d[1] * d[2] - s12
    a = jnp.sum(d) + A
    b = d[0] * d[1] - s01 + d[0] * d[2] - s02 + c_ee + A * (d[1] + d[2])
    det = jnp.real(jnp.linalg.det(h0)) + A * c_ee
    p = jnp.maximum(a * a / 3.0 - b, 0.0)
    q = 2.0 * a ** 3 / 27.0 - a * b / 3.0 + det
    r = jnp.sqrt(p / 3.0)
    phi = jnp.arccos(jnp.clip(q / (2.0 * r ** 3), -1.0, 1.0)) / 3.0
    lam = jnp.stack([a / 3.0 + 2.0 * r * jnp.cos(phi + 2.0 * np.pi / 3.0),
                     a / 3.0 + 2.0 * r * jnp.cos(phi - 2.0 * np.pi / 3.0),
                     a / 3.0 + 2.0 * r * jnp.cos(phi)], axis=-1)
    i, j = [x for x in range(3) if x != alpha]
    hd = jnp.stack([d[0] + A, jnp.full_like(A, d[1]), jnp.full_like(A, d[2])], axis=-1)
    cof = (lam - hd[:, i, None]) * (lam - hd[:, j, None]) - jnp.abs(h0[i, j]) ** 2
    den = jnp.stack([(lam[:, 0] - lam[:, 1]) * (lam[:, 0] - lam[:, 2]),
                     (lam[:, 1] - lam[:, 0]) * (lam[:, 1] - lam[:, 2]),
                     (lam[:, 2] - lam[:, 0]) * (lam[:, 2] - lam[:, 1])], axis=-1)
    h = cof / den
    corr = jnp.fft.irfft(jnp.fft.rfft(h.T, n_fft) * kernel_fft, n_fft)[:, m - 1:n_pad]
    grid = u_lo + _DU * np.arange(n_out)
    out = jnp.stack([jnp.interp(jnp.asarray(u_e), jnp.asarray(grid), corr[c]) for c in range(3)], axis=-1)
    return out.reshape(energy.shape + (3,))


def solar_probabilities(params, energy_GeV, production, source="8B", density_gcc=None, ye=None,
                        length_km=None, anti=False, nsi=None, method="sum", backend="cayley"):
    """``P(nu_e -> nu_beta)`` for solar neutrinos of ``source``, by day or at night.

    Neutrinos produced as nu_e reach the Earth as an incoherent mixture of mass
    eigenstates with the fractions of :func:`production_mixing`. By day (no
    segments) ``P[..., beta] = sum_k c_k |U_{beta k}|^2``, shape ``([n_E,] N)``. At
    night, ``density_gcc``, ``ye`` and ``length_km`` describe the chord through the
    Earth, ordered from entry to detector, 1-D or 2-D ``(n_paths, n_seg)`` as in
    :func:`mango.amplitude_profile`, and ``P[..., beta] = sum_k c_k P(nu_k -> nu_beta)``
    with shape ``([n_paths,] [n_E,] N)``.
    """
    from .oscillator import mass_probability_profile

    c = production_mixing(params, energy_GeV, production, source=source, alpha=0, anti=anti,
                          nsi=nsi, method=method)
    if density_gcc is None:
        u = params.pmns()
        return c @ (jnp.abs(jnp.conj(u) if anti else u) ** 2).T
    p = mass_probability_profile(params, energy_GeV, density_gcc, ye, length_km, anti=anti,
                                 backend=backend, nsi=nsi)
    return jnp.einsum("...bk,...k->...b", p, c)
