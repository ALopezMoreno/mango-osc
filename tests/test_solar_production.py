"""Tests for neutron densities on solar profiles and production-averaged adiabatic mixing."""

import dataclasses
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
