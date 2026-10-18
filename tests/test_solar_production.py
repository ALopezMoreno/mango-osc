"""Tests for the bundled solar production data, production-averaged mixing and solar day/night probabilities."""

import dataclasses
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import Sterile3plus1, NSI, nufit_no, mass_probability_profile, solar
from mango import constants as C

ENERGY = np.linspace(3e-3, 15e-3, 9)          # GeV
RHO = np.array([3.4, 5.3, 10.5, 12.2, 10.5, 5.3, 3.4])
YE = np.array([0.496, 0.496, 0.466, 0.466, 0.466, 0.496, 0.496])
LENGTH = np.array([600.0, 2200.0, 1300.0, 2400.0, 1300.0, 2200.0, 600.0])


def _brute_mixing(params, energy_GeV, production, source, alpha=0, nsi=None):
    """sum_r f(r) |<nu_k^m(r)|nu_alpha>|^2 with numpy eigh of a Hamiltonian built here, radius by radius."""
    u, msq = np.asarray(params.pmns()), np.asarray(params.msquared())
    w = np.asarray(production.weights[source])
    keep = w > 1e-14 * w.max()
    n_e, w = np.asarray(production.n_e)[keep], w[keep] / w[keep].sum()
    out = []
    for e in energy_GeV:
        e_eV = e * C.GEV_TO_EV
        h0 = (u * msq / (2 * e_eV)) @ np.conj(u).T
        m = np.diag([1.0, 0.0, 0.0]) + (0 if nsi is None else nsi)
        h = h0[None] + C.V_FACTOR_EV * n_e[:, None, None] * m[None]
        out.append(w @ (np.abs(np.linalg.eigh(h)[1][:, alpha, :]) ** 2))
    return np.array(out)


def test_load_production():
    for model in solar.SOLAR_MODELS:
        prod = solar.load_production(model)
        for s in solar.SOURCES:
            assert np.sum(prod.weights[s]) == pytest.approx(1.0, abs=1e-10)
        # helium-rich core: about one neutron per two electrons
        assert 0.47 < prod.n_n[0] / prod.n_e[0] < 0.50


def test_production_mixing_against_brute_force():
    p, prod = nufit_no(), solar.load_production("SF3_GS98")
    for alpha in (0, 1):
        got = np.asarray(solar.production_mixing(p, ENERGY, prod, "8B", alpha=alpha))
        assert np.max(np.abs(got - _brute_mixing(p, ENERGY, prod, "8B", alpha))) < 1e-10
        assert np.max(np.abs(got.sum(axis=-1) - 1.0)) < 1e-12


def test_convolution_fast_path():
    p, prod = nufit_no(), solar.load_production("BS05_OP")
    for source in ("8B", "pp", "7Be"):
        for alpha in (0, 2):
            exact = solar.production_mixing(p, ENERGY, prod, source, alpha=alpha)
            fast = solar.production_mixing(p, ENERGY, prod, source, alpha=alpha, method="convolution")
            assert np.max(np.abs(np.asarray(fast - exact))) < 1e-7
    with pytest.raises(ValueError):
        solar.production_mixing(p, ENERGY, prod, nsi=NSI(eps_ee=0.1), method="convolution")


def test_two_flavour_closed_form():
    """With theta13 = 0 the electron row lives in the (1, 2) block, and the adiabatic result is exact."""
    p = dataclasses.replace(nufit_no(), theta13=jnp.asarray(0.0))
    prod = solar.load_production("SF3_GS98")
    w = np.asarray(prod.weights["8B"])
    th, dm = float(p.theta12), float(p.dm21)
    for e in ENERGY:
        a = 2 * e * C.GEV_TO_EV * C.V_FACTOR_EV * np.asarray(prod.n_e)
        c2m = (dm * np.cos(2 * th) - a) / np.hypot(dm * np.cos(2 * th) - a, dm * np.sin(2 * th))
        ref = w @ (0.5 * (1 + np.cos(2 * th) * c2m))
        got = float(solar.solar_probabilities(p, e, prod)[0])
        assert got == pytest.approx(ref, abs=1e-12)


def test_nsi_mixing_against_brute_force():
    p, prod = nufit_no(), solar.load_production("SF3_GS98")
    nsi = NSI(eps_ee=0.1, eps_emu=0.05 + 0.02j, eps_tautau=-0.05)
    got = np.asarray(solar.production_mixing(p, ENERGY, prod, "8B", nsi=nsi))
    ref = _brute_mixing(p, ENERGY, prod, "8B", nsi=np.asarray(nsi.matrix(3)))
    assert np.max(np.abs(got - ref)) < 1e-10


def test_sterile_production_mixing():
    p3, prod = nufit_no(), solar.load_production("SF3_GS98")

    def sterile(theta14):
        return Sterile3plus1(theta12=p3.theta12, theta13=p3.theta13, theta23=p3.theta23,
                             theta14=jnp.asarray(theta14), theta24=jnp.asarray(0.0), theta34=jnp.asarray(0.0),
                             delta13=p3.deltacp, delta24=jnp.asarray(0.0), dm21=p3.dm21, dm31=p3.dm31,
                             dm41=jnp.asarray(1.0))

    c4 = np.asarray(solar.production_mixing(sterile(0.0), ENERGY, prod))
    c3 = np.asarray(solar.production_mixing(p3, ENERGY, prod))
    assert np.max(np.abs(c4[:, :3] - c3)) < 1e-10 and np.max(np.abs(c4[:, 3])) < 1e-12
    c4 = np.asarray(solar.production_mixing(sterile(0.1), ENERGY, prod))
    assert np.max(np.abs(c4.sum(axis=-1) - 1.0)) < 1e-12 and np.min(c4[:, 3]) > 0


def test_day_and_night():
    p, prod = nufit_no(), solar.load_production("SF3_GS98")
    day = np.asarray(solar.solar_probabilities(p, ENERGY, prod))
    assert np.max(np.abs(day.sum(axis=-1) - 1.0)) < 1e-12
    # an empty chord changes nothing
    empty = np.asarray(solar.solar_probabilities(p, ENERGY, prod, density_gcc=np.zeros(2), ye=0.5,
                                                 length_km=np.zeros(2)))
    assert np.max(np.abs(empty - day)) < 1e-12
    night = np.asarray(solar.solar_probabilities(p, ENERGY, prod, density_gcc=RHO, ye=YE, length_km=LENGTH))
    c = np.asarray(solar.production_mixing(p, ENERGY, prod))
    pm = np.asarray(mass_probability_profile(p, ENERGY, RHO, YE, LENGTH))
    assert np.max(np.abs(night - np.einsum("ebk,ek->eb", pm, c))) < 1e-14
    # Earth regeneration raises P_ee at night on average; on one core-crossing chord it reaches several percent
    assert np.mean(night[:, 0] - day[:, 0]) > 0 and np.max(np.abs(night[:, 0] - day[:, 0])) < 0.1
    # a batch of two chords
    batch = np.asarray(solar.solar_probabilities(p, ENERGY, prod, density_gcc=np.stack([RHO, RHO]),
                                                 ye=np.stack([YE, YE]), length_km=np.stack([LENGTH, LENGTH])))
    assert batch.shape == (2, len(ENERGY), 3) and np.max(np.abs(batch[1] - night)) < 1e-12


def test_gradients():
    prod = solar.load_production("SF3_GS98")

    def f(dm21, method):
        p = dataclasses.replace(nufit_no(), dm21=dm21)
        day = solar.solar_probabilities(p, ENERGY, prod, method=method)[:, 0]
        night = solar.solar_probabilities(p, ENERGY, prod, density_gcc=RHO, ye=YE, length_km=LENGTH,
                                          method=method)[:, 0]
        return jnp.sum(day) + 10.0 * jnp.sum(night - day)

    dm = jnp.asarray(7.4e-5)
    for method in ("sum", "convolution"):
        ad = float(jax.grad(f)(dm, method))
        h = 1e-10
        fd = (float(f(dm + h, method)) - float(f(dm - h, method))) / (2 * h)
        assert ad == pytest.approx(fd, rel=1e-5)


def test_bs05_profile_neutrons_and_sterile_fractions():
    prof = solar.load_bs05(os.path.join(os.path.dirname(__file__), "..", "examples", "data", "bs05_agsop.dat"))
    assert prof.rho_yn is not None
    p3 = nufit_no()
    st = Sterile3plus1(theta12=p3.theta12, theta13=p3.theta13, theta23=p3.theta23, theta14=jnp.asarray(0.1),
                       theta24=jnp.asarray(0.0), theta34=jnp.asarray(0.0), delta13=p3.deltacp,
                       delta24=jnp.asarray(0.0), dm21=p3.dm21, dm31=p3.dm31, dm41=jnp.asarray(1.0))
    F = np.asarray(solar.adiabatic_mass_fractions(st, 8e-3, prof, prof.R_sun_km, 0.05 * prof.R_sun_km))
    assert F.shape == (4,) and F.sum() == pytest.approx(1.0, abs=1e-12)
