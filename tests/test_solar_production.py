"""Tests for neutron densities on solar profiles and production-averaged adiabatic mixing."""

import dataclasses
import warnings
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import NSI, Sterile3plus1, nufit_no, probability_solar, solar
from mango import constants as C

PROFILE = solar.load_bs05(os.path.join(os.path.dirname(__file__), "..", "examples", "data", "bs05_agsop.dat"))
R = np.asarray(PROFILE.r_over_rsun)
WEIGHTS = np.exp(-0.5 * ((R - 0.045) / 0.015) ** 2)   # an 8B-like production region
ENERGY = np.linspace(3e-3, 15e-3, 9)                    # GeV


def _inverted(p):
    return dataclasses.replace(p, dm31=-p.dm31 + p.dm21)


def _sterile(theta14=0.15):
    p = nufit_no()
    return Sterile3plus1(theta12=p.theta12, theta13=p.theta13, theta23=p.theta23,
                         theta14=jnp.asarray(theta14), theta24=jnp.asarray(0.1), theta34=jnp.asarray(0.0),
                         delta13=p.deltacp, delta24=jnp.asarray(0.0), dm21=p.dm21, dm31=p.dm31,
                         dm41=jnp.asarray(1e-4))


def _brute(params, alpha=0, anti=False, neutrons=False):
    """Production-averaged mass-state fractions with numpy eigh, radius by radius.

    Each matter eigenstate is assigned to the mass state its vacuum limit overlaps most, so no
    ordering rule is assumed.
    """
    u, msq = np.asarray(params.pmns()), np.asarray(params.msquared())
    n = len(msq)
    na = getattr(params, "n_active", n)
    if anti:
        u = np.conj(u)
    keep = WEIGHTS > 1e-14 * WEIGHTS.max()
    w = WEIGHTS[keep] / WEIGHTS[keep].sum()
    sign = -1.0 if anti else 1.0
    v_cc = sign * C.V_FACTOR_EV * np.asarray(PROFILE.rho_ye)[keep]
    v_nc = sign * 0.5 * C.V_FACTOR_EV * np.asarray(PROFILE.rho_yn)[keep] if neutrons else 0 * v_cc
    m_cc = np.zeros((n, n)); m_cc[0, 0] = 1.0
    m_nc = np.diag((np.arange(n) >= na).astype(float))
    out = []
    for e in ENERGY:
        h0 = (u * msq / (2 * e * C.GEV_TO_EV)) @ np.conj(u).T
        vec = np.linalg.eigh(h0[None] + v_cc[:, None, None] * m_cc + v_nc[:, None, None] * m_nc)[1]
        perm = np.argmax(np.abs(np.conj(u).T @ np.linalg.eigh(h0)[1]) ** 2, axis=0)
        c = np.zeros(n)
        c[perm] = w @ (np.abs(vec[:, alpha, :]) ** 2)
        out.append(c)
    return np.array(out)


def test_neutron_density():
    # helium-rich core: about one neutron per two electrons; hydrogen-rich surface: fewer
    ratio = np.asarray(PROFILE.rho_yn) / np.asarray(PROFILE.rho_ye)
    assert 0.45 < ratio[0] < 0.50
    assert ratio[-1] < ratio[0]
    assert solar.exponential_profile().rho_yn is None


@pytest.mark.parametrize("params", [nufit_no(), _inverted(nufit_no())], ids=["NO", "IO"])
def test_sum_against_brute_force(params):
    for alpha in (0, 1):
        got = np.asarray(solar.production_mixing(params, ENERGY, PROFILE, WEIGHTS, alpha=alpha))
        assert np.max(np.abs(got - _brute(params, alpha))) < 1e-12
        assert np.max(np.abs(got.sum(axis=-1) - 1.0)) < 1e-12


def test_antineutrinos():
    p = nufit_no()
    got = np.asarray(solar.production_mixing(p, ENERGY, PROFILE, WEIGHTS, anti=True))
    assert np.max(np.abs(got - _brute(p, anti=True))) < 1e-12
    # adiabatic_mass_fractions at the surface, from one emission radius, against the same construction
    r_sun = PROFILE.R_sun_km
    f = np.asarray(solar.adiabatic_mass_fractions(p, 0.01, PROFILE, 0.999 * r_sun, 0.05 * r_sun, anti=True))
    u, msq = np.conj(np.asarray(p.pmns())), np.asarray(p.msquared())

    def vecs(r_km):
        v = -float(solar.potential_eV(PROFILE, r_km))
        return np.linalg.eigh((u * msq / (2e7)) @ np.conj(u).T + np.diag([v, 0.0, 0.0]))[1]

    ref = (np.abs(np.conj(u).T @ vecs(0.999 * r_sun)) ** 2) @ (np.abs(vecs(0.05 * r_sun)[0]) ** 2)
    assert np.max(np.abs(f - ref)) < 1e-12


def test_steriles_feel_neutrons():
    st = _sterile()
    got = np.asarray(solar.production_mixing(st, ENERGY, PROFILE, WEIGHTS))
    assert np.max(np.abs(got - _brute(st, neutrons=True))) < 1e-12
    blind = PROFILE._replace(rho_yn=None)
    assert np.max(np.abs(got - np.asarray(solar.production_mixing(st, ENERGY, blind, WEIGHTS)))) > 1e-6
    with pytest.raises(ValueError):
        solar.production_mixing(st, ENERGY, PROFILE, WEIGHTS, method="convolution")


@pytest.mark.parametrize("kw", [dict(), dict(alpha=2), dict(anti=True), dict(nsi=NSI(eps_ee=0.3, eps_etau=0.1)),
                                dict(params=_inverted(nufit_no()))], ids=["NO", "alpha", "anti", "NSI", "IO"])
def test_convolution_matches_sum(kw):
    params = kw.pop("params", nufit_no())
    exact = solar.production_mixing(params, ENERGY, PROFILE, WEIGHTS, **kw)
    fast = solar.production_mixing(params, ENERGY, PROFILE, WEIGHTS, method="convolution", **kw)
    assert np.max(np.abs(np.asarray(fast - exact))) < 1e-7


def test_gradients():
    p = nufit_no()

    def f(dm21, method):
        q = dataclasses.replace(p, dm21=dm21)
        return solar.production_mixing(q, ENERGY, PROFILE, WEIGHTS, method=method)[:, 0].sum()

    dm21, h = p.dm21, 1e-9
    fd = (float(f(dm21 + h, "sum")) - float(f(dm21 - h, "sum"))) / (2 * h)
    for method in ("sum", "convolution"):
        assert float(jax.grad(f)(dm21, method)) == pytest.approx(fd, rel=1e-5)


def _night_reference(params, c, cz):
    """sum_i c_i |(S U)_{beta i}|^2 with S from numpy eigendecompositions along MANGO's chord."""
    from mango import earth
    rho, ye, length = (np.asarray(x) for x in earth.chord_segments(jnp.asarray(cz), earth.shell_table(4)))
    u, msq = np.asarray(params.pmns()), np.asarray(params.msquared())
    out = []
    for e, ce in zip(ENERGY, c):
        s = np.eye(3, dtype=complex)
        for r, y, l in zip(rho, ye, length):
            h = (u * msq / (2 * e * C.GEV_TO_EV)) @ np.conj(u).T + np.diag([C.V_FACTOR_EV * r * y, 0.0, 0.0])
            w, vec = np.linalg.eigh(h)
            s = (vec * np.exp(-1j * w * l * C.KM_TO_INV_EV)) @ np.conj(vec).T @ s
        out.append((np.abs(s @ u) ** 2) @ ce)
    return np.array(out)


@pytest.mark.parametrize("params", [nufit_no(), _inverted(nufit_no())], ids=["NO", "IO"])
def test_probability_solar(params):
    c = _brute(params)
    day = np.asarray(probability_solar(params, ENERGY, PROFILE, WEIGHTS))
    assert np.max(np.abs(day - c @ np.abs(np.asarray(params.pmns())).T ** 2)) < 1e-12
    assert np.max(np.abs(day.sum(axis=-1) - 1.0)) < 1e-12
    night = np.asarray(probability_solar(params, ENERGY, PROFILE, WEIGHTS, cos_zenith=np.array([-0.9, -0.3])))
    assert night.shape == (2, len(ENERGY), 3)
    for i, cz in enumerate((-0.9, -0.3)):
        assert np.max(np.abs(night[i] - _night_reference(params, c, cz))) < 1e-11
    pee = probability_solar(params, ENERGY, PROFILE, WEIGHTS, cos_zenith=-0.3, flavor_out=0, method="convolution")
    assert np.max(np.abs(np.asarray(pee) - night[1, :, 0])) < 1e-7


# --- the adiabatic approximation against exact propagation --------------------

def _sterile4(theta14, dm41, theta24=0.0):
    p = nufit_no()
    return Sterile3plus1(theta12=p.theta12, theta13=p.theta13, theta23=p.theta23,
                         theta14=jnp.asarray(theta14), theta24=jnp.asarray(theta24), theta34=jnp.asarray(0.0),
                         delta13=p.deltacp, delta24=jnp.asarray(0.0), dm21=p.dm21, dm31=p.dm31,
                         dm41=jnp.asarray(dm41))


def _smooth_profile(n=20001):
    """Bahcall's exponential n_e, with n_n = 0.4 n_e, finely tabulated."""
    r = np.linspace(0.0, 1.0, n)
    ne = 245.0 * np.exp(-10.54 * r)
    return solar.SolarProfile(r, ne, solar.R_SUN_KM, 0.4 * ne), (lambda x: 245.0 * np.exp(-10.54 * x))


def _exact_fractions(params, energy_GeV, density, r0, n_layers=40000, neutrons=0.4, anti=False):
    """|(U^dag S)_{i e}|^2 for nu_e propagated through thin layers from r0 [R_sun] to the surface."""
    from mango.layers import propagate_layers
    edges = np.linspace(r0, 1.0, n_layers + 1)
    ne = density(0.5 * (edges[1:] + edges[:-1]))
    u, msq = params.pmns(), params.msquared()
    s = propagate_layers(u, msq, energy_GeV * C.GEV_TO_EV, C.matter_potential_eV(jnp.asarray(ne), 1.0),
                         jnp.asarray(np.diff(edges) * solar.R_SUN_KM * C.KM_TO_INV_EV),
                         v_nc=0.5 * C.matter_potential_eV(jnp.asarray(neutrons * ne), 1.0), anti=anti,
                         n_active=getattr(params, "n_active", 3), backend="eigh")
    u_in = np.conj(np.asarray(u)) if anti else np.asarray(u)
    return np.abs(np.conj(u_in).T @ np.asarray(s)[:, 0]) ** 2


def _at(profile, r0):
    w = np.zeros(len(profile.r_over_rsun))
    w[np.argmin(np.abs(np.asarray(profile.r_over_rsun) - r0))] = 1.0
    return w


def test_adiabatic_against_exact_propagation_bs05():
    """3 flavors from r = 0.05 through the BS05 table: adiabatic and exact agree to ~3e-5."""
    p, r = nufit_no(), np.asarray(PROFILE.r_over_rsun)
    w = _at(PROFILE, 0.05)
    exact = _exact_fractions(p, 0.01, lambda x: np.interp(x, r, np.asarray(PROFILE.rho_ye)), r[np.argmax(w)])
    assert np.max(np.abs(np.asarray(solar.production_mixing(p, 0.01, PROFILE, w)) - exact)) < 1e-4


@pytest.mark.parametrize("theta14, energy, tol", [(0.1, 0.01, 1e-4), (0.05, 0.01, 1e-4)])
def test_adiabatic_steriles_against_exact_propagation(theta14, energy, tol):
    """Adiabatic sterile resonances (gamma ~ 10-50) on a smooth profile: no warning, exact to ~1e-5."""
    profile, density = _smooth_profile()
    p, w = _sterile4(theta14, 1e-4), _at(profile, 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = np.asarray(solar.production_mixing(p, energy, profile, w))
    assert np.max(np.abs(got - _exact_fractions(p, energy, density, 0.05, neutrons=0.4))) < tol


def test_nonadiabatic_steriles_warn():
    """theta14 = 0.02 at 1e-4 eV^2, 10 MeV: gamma ~ 1.8, Landau-Zener jumps, and a warning."""
    profile, density = _smooth_profile()
    p, w = _sterile4(0.02, 1e-4), _at(profile, 0.05)
    with pytest.warns(UserWarning, match="adiabaticity"):
        got = np.asarray(solar.production_mixing(p, 0.01, profile, w))
    assert np.max(np.abs(got - _exact_fractions(p, 0.01, density, 0.05, neutrons=0.4))) > 1e-2


def test_adiabaticity_two_flavor():
    """Two flavors, exponential profile: gamma = a D^3 / (beta sin 2theta |d ln n_e/dx|), minimal at
    beta = (cos 2theta + sqrt(cos^2 2theta + 8)) / 4, just past the resonance."""
    p = dataclasses.replace(nufit_no(), theta13=0.0)
    profile, _ = _smooth_profile()
    c2, s2, e = np.cos(2 * float(p.theta12)), np.sin(2 * float(p.theta12)), 0.01
    a = float(p.dm21) / (2 * e * C.GEV_TO_EV)
    slope = 10.54 / (solar.R_SUN_KM * C.KM_TO_INV_EV)           # |d ln n_e / dx| [eV]
    beta = (c2 + np.sqrt(c2 ** 2 + 8)) / 4
    expected = a * ((c2 - beta) ** 2 + s2 ** 2) ** 1.5 / (beta * s2 * slope)
    assert float(solar.adiabaticity(p, e, profile)) == pytest.approx(expected, rel=1e-3)


# --- inputs ---------------------------------------------------------------------

def test_weight_validation():
    p = nufit_no()
    for bad, match in ((WEIGHTS[:-1], "one entry per profile radius"), (-WEIGHTS, "non-negative"),
                       (0 * WEIGHTS, "not all be zero")):
        with pytest.raises(ValueError, match=match):
            solar.production_mixing(p, ENERGY, PROFILE, bad)


def test_profile_gradient():
    """method='sum' is differentiable in the solar model and in the weights."""
    p, e = nufit_no(), 0.008

    def c_e(scale, tilt):
        prof = PROFILE._replace(rho_ye=scale * jnp.asarray(PROFILE.rho_ye))
        return solar.production_mixing(p, e, prof, jnp.asarray(WEIGHTS) * jnp.exp(tilt * jnp.asarray(R)))[1]

    g = jax.grad(c_e, argnums=(0, 1))(1.0, 0.0)
    h = 1e-5
    for i, fd in enumerate(((c_e(1 + h, 0.0) - c_e(1 - h, 0.0)) / (2 * h), (c_e(1.0, h) - c_e(1.0, -h)) / (2 * h))):
        assert float(g[i]) == pytest.approx(float(fd), rel=1e-5)
    with pytest.raises(ValueError, match="concrete"):
        jax.grad(lambda s: solar.production_mixing(p, e, PROFILE._replace(rho_ye=s * jnp.asarray(PROFILE.rho_ye)),
                                                   WEIGHTS, method="convolution")[1])(1.0)


def test_probability_solar_options():
    p = nufit_no()
    with pytest.raises(TypeError, match="cos_zenith"):
        probability_solar(p, ENERGY, PROFILE, WEIGHTS, det_depth_km=1.0)
    mu = np.asarray(probability_solar(p, ENERGY, PROFILE, WEIGHTS, alpha=1))
    c = np.asarray(solar.production_mixing(p, ENERGY, PROFILE, WEIGHTS, alpha=1))
    assert np.max(np.abs(mu - c @ np.abs(np.asarray(p.pmns())).T ** 2)) < 1e-12


def _exact_region(params, energy, density, radii, weights, anti=False):
    """Exact fractions averaged incoherently over production radii."""
    out = [w * _exact_fractions(params, energy, density, r, n_layers=20000, anti=anti) for r, w in zip(radii, weights)]
    return np.sum(out, axis=0) / np.sum(weights)


def test_outer_exact_steriles():
    """Parke factorisation: a resonance outside the production region, gamma ~ 0.4, from 0.47 to < 1e-3."""
    profile, density = _smooth_profile()
    r = np.asarray(profile.r_over_rsun)
    radii = np.linspace(0.0, 0.1, 21)
    shape = np.exp(-0.5 * ((radii - 0.045) / 0.015) ** 2)
    weights = np.interp(r, radii, shape) * (r <= 0.1)
    p = _sterile4(0.01, 1e-4)
    exact = _exact_region(p, 0.01, density, radii, shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = np.asarray(solar.production_mixing(p, 0.01, profile, weights, outer="exact"))
    assert np.max(np.abs(got - exact)) < 1e-3
    three, e = nufit_no(), np.array([0.006, 0.01])
    exact3 = np.asarray(solar.production_mixing(three, e, profile, weights, outer="exact"))
    assert np.max(np.abs(exact3 - np.asarray(solar.production_mixing(three, e, profile, weights)))) < 1e-4
    conv3 = solar.production_mixing(three, e, profile, weights, outer="exact", method="convolution")
    assert np.max(np.abs(np.asarray(conv3) - exact3)) < 1e-7


def test_outer_exact_warns_inside_production_region():
    """For antineutrinos the sterile resonance lies inside the production region: outer='exact' still warns."""
    profile, _ = _smooth_profile()
    r = np.asarray(profile.r_over_rsun)
    weights = np.exp(-0.5 * ((r - 0.045) / 0.015) ** 2) * (r <= 0.1)
    with pytest.warns(UserWarning, match="inside the production region"):
        solar.production_mixing(_sterile4(0.01, 1e-4), 0.01, profile, weights, anti=True, outer="exact")
