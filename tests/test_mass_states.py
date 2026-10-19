"""Tests for ``mass_states=True``: vacuum mass eigenstates entering a profile or the Earth."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import Sterile3plus1, nufit_no, probability_earth, probability_profile
from mango import constants as C
from mango import earth as E

# A PREM-like chord through the mantle and core, source -> detector, and solar-neutrino energies.
RHO = np.array([3.4, 4.4, 5.3, 10.5, 12.2, 10.5, 5.3, 4.4, 3.4])
YE = np.array([0.496, 0.496, 0.496, 0.466, 0.466, 0.466, 0.496, 0.496, 0.496])
LENGTH = np.array([400.0, 900.0, 1500.0, 1200.0, 2100.0, 1200.0, 1500.0, 900.0, 400.0])
ENERGY = np.linspace(4e-3, 15e-3, 7)


def _reference(params, energy_GeV, rho, ye, length_km, anti=False):
    """|(S U)_{beta i}|^2 from numpy eigendecompositions of the full Hamiltonian, segment by segment."""
    u = np.asarray(params.pmns())
    if anti:
        u = np.conj(u)
    msq = np.asarray(params.msquared())
    out = []
    for e in energy_GeV:
        e_eV = e * C.GEV_TO_EV
        s = np.eye(3, dtype=complex)
        for r, y, l in zip(rho, ye, length_km):
            v = C.V_FACTOR_EV * r * y * (-1 if anti else 1)
            h = (u * msq / (2 * e_eV)) @ np.conj(u).T + np.diag([v, 0.0, 0.0])
            w, vec = np.linalg.eigh(h)
            s = (vec * np.exp(-1j * w * l * C.KM_TO_INV_EV)) @ np.conj(vec).T @ s
        out.append(np.abs(s @ u) ** 2)
    return np.array(out)


@pytest.mark.parametrize("anti", [False, True])
def test_profile_against_independent_exponentials(anti):
    p = nufit_no()
    got = np.asarray(probability_profile(p, ENERGY, RHO, YE, LENGTH, anti=anti, mass_states=True))
    assert np.max(np.abs(got - _reference(p, ENERGY, RHO, YE, LENGTH, anti=anti))) < 1e-12
    assert np.max(np.abs(got.sum(axis=-2) - 1.0)) < 1e-12


def test_vacuum_limit():
    p = nufit_no()
    got = np.asarray(probability_profile(p, ENERGY, np.zeros(3), 0.5, np.array([1e3, 5e3, 7e3]),
                                         mass_states=True))
    assert np.max(np.abs(got - np.abs(np.asarray(p.pmns())) ** 2)) < 1e-12


def test_earth_against_independent_exponentials():
    p = nufit_no()
    for cz in (-0.95, -0.4):
        rho, ye, length = (np.asarray(x) for x in E.chord_segments(jnp.asarray(cz), E.shell_table(4)))
        got = np.asarray(probability_earth(p, ENERGY, cz, mass_states=True))
        assert np.max(np.abs(got - _reference(p, ENERGY, rho, ye, length))) < 1e-11
    grid = np.asarray(probability_earth(p, ENERGY, np.array([-0.95, -0.4]), mass_states=True))
    assert grid.shape == (2, len(ENERGY), 3, 3)


def test_selected_entry_and_gradient():
    p = nufit_no()
    full = probability_profile(p, ENERGY, RHO, YE, LENGTH, mass_states=True)
    p1e = probability_profile(p, ENERGY, RHO, YE, LENGTH, mass_states=True, flavor_in=0, flavor_out=0)
    assert np.max(np.abs(np.asarray(full[..., 0, 0] - p1e))) < 1e-15

    def f(rho):
        return jnp.sum(probability_profile(p, ENERGY, rho, YE, LENGTH, mass_states=True,
                                           flavor_in=0, flavor_out=0))

    rho = jnp.asarray(RHO)
    v = np.linspace(-1.0, 1.0, len(RHO))
    h = 1e-5
    fd = (float(f(rho + h * v)) - float(f(rho - h * v))) / (2 * h)
    assert np.asarray(jax.grad(f)(rho)) @ v == pytest.approx(fd, rel=1e-6)


def test_sterile_mass_states():
    p3 = nufit_no()
    st = Sterile3plus1(theta12=p3.theta12, theta13=p3.theta13, theta23=p3.theta23,
                       theta14=jnp.asarray(0.15), theta24=jnp.asarray(0.1), theta34=jnp.asarray(0.0),
                       delta13=p3.deltacp, delta24=jnp.asarray(0.0), dm21=p3.dm21, dm31=p3.dm31,
                       dm41=jnp.asarray(1.0))
    got = np.asarray(probability_profile(st, ENERGY, RHO, YE, LENGTH, mass_states=True))
    assert got.shape == (len(ENERGY), 4, 4)
    assert np.max(np.abs(got.sum(axis=-2) - 1.0)) < 1e-10
