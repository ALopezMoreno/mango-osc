"""Tests for amplitude_profile and mass_probability_profile (mass eigenstates entering a profile)."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import (Sterile3plus1, nufit_no, amplitude_profile, mass_probability_profile,
                   probability_profile, Flavor)
from mango import constants as C

# A PREM-like chord through the mantle and core, source -> detector, and solar-neutrino energies.
RHO = np.array([3.4, 4.4, 5.3, 10.5, 12.2, 10.5, 5.3, 4.4, 3.4])
YE = np.array([0.496, 0.496, 0.496, 0.466, 0.466, 0.466, 0.496, 0.496, 0.496])
LENGTH = np.array([400.0, 900.0, 1500.0, 1200.0, 2100.0, 1200.0, 1500.0, 900.0, 400.0])
ENERGY = np.linspace(4e-3, 15e-3, 7)


def _reference_mass_probabilities(params, energy_GeV, rho, ye, length_km, anti=False):
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


def test_amplitude_unitary_and_consistent_with_probability_profile():
    p = nufit_no()
    s = amplitude_profile(p, ENERGY, RHO, YE, LENGTH)
    assert s.shape == (len(ENERGY), 3, 3)
    eye = np.eye(3)
    assert np.max(np.abs(np.asarray(s @ jnp.conj(jnp.swapaxes(s, -1, -2))) - eye)) < 1e-12
    p_flavor = probability_profile(p, ENERGY, RHO, YE, LENGTH)
    assert np.max(np.abs(np.asarray(jnp.abs(s) ** 2 - p_flavor))) < 1e-14


@pytest.mark.parametrize("anti", [False, True])
def test_mass_probabilities_against_independent_exponentials(anti):
    p = nufit_no()
    got = np.asarray(mass_probability_profile(p, ENERGY, RHO, YE, LENGTH, anti=anti))
    ref = _reference_mass_probabilities(p, ENERGY, RHO, YE, LENGTH, anti=anti)
    assert np.max(np.abs(got - ref)) < 1e-12
    # each mass state ends up somewhere: columns sum to one
    assert np.max(np.abs(got.sum(axis=-2) - 1.0)) < 1e-12


def test_mass_probabilities_vacuum_limit():
    p = nufit_no()
    got = np.asarray(mass_probability_profile(p, ENERGY, np.zeros(3), 0.5, np.array([1e3, 5e3, 7e3])))
    assert np.max(np.abs(got - np.abs(np.asarray(p.pmns())) ** 2)) < 1e-12


def test_batched_paths_with_zero_length_padding():
    p = nufit_no()
    short = slice(2, 7)
    rho = np.stack([RHO, np.concatenate([np.zeros(4), RHO[short]])])
    ye = np.stack([YE, np.concatenate([np.full(4, 0.5), YE[short]])])
    length = np.stack([LENGTH, np.concatenate([np.zeros(4), LENGTH[short]])])
    batch = np.asarray(mass_probability_profile(p, ENERGY, rho, ye, length))
    assert batch.shape == (2, len(ENERGY), 3, 3)
    one = np.asarray(mass_probability_profile(p, ENERGY, RHO, YE, LENGTH))
    two = np.asarray(mass_probability_profile(p, ENERGY, RHO[short], YE[short], LENGTH[short]))
    assert np.max(np.abs(batch[0] - one)) < 1e-12
    assert np.max(np.abs(batch[1] - two)) < 1e-12


def test_selected_entry_and_gradient():
    p = nufit_no()
    full = mass_probability_profile(p, ENERGY, RHO, YE, LENGTH)
    p1e = mass_probability_profile(p, ENERGY, RHO, YE, LENGTH, mass_in=0, flavor_out=Flavor.E)
    assert np.max(np.abs(np.asarray(full[..., 0, 0] - p1e))) < 1e-15

    def f(rho):
        return jnp.sum(mass_probability_profile(p, ENERGY, rho, YE, LENGTH, mass_in=0, flavor_out=Flavor.E))

    rho = jnp.asarray(RHO)
    grad = np.asarray(jax.grad(f)(rho))
    v = np.linspace(-1.0, 1.0, len(RHO))
    h = 1e-5
    fd = (float(f(rho + h * v)) - float(f(rho - h * v))) / (2 * h)
    assert grad @ v == pytest.approx(fd, rel=1e-6)


def test_sterile_mass_probabilities():
    p3 = nufit_no()
    st = Sterile3plus1(theta12=p3.theta12, theta13=p3.theta13, theta23=p3.theta23,
                       theta14=jnp.asarray(0.15), theta24=jnp.asarray(0.1), theta34=jnp.asarray(0.0),
                       delta13=p3.deltacp, delta24=jnp.asarray(0.0), dm21=p3.dm21, dm31=p3.dm31,
                       dm41=jnp.asarray(1.0))
    got = np.asarray(mass_probability_profile(st, ENERGY, RHO, YE, LENGTH))
    assert got.shape == (len(ENERGY), 4, 4)
    assert np.max(np.abs(got.sum(axis=-2) - 1.0)) < 1e-10


def test_input_validation():
    p = nufit_no()
    with pytest.raises(ValueError):
        amplitude_profile(p, ENERGY, np.ones((2, 2, 3)), 0.5, np.ones((2, 2, 3)))
    with pytest.raises(ValueError):
        amplitude_profile(p, ENERGY, RHO, YE, -LENGTH)
    with pytest.raises(ValueError):
        mass_probability_profile(p, ENERGY, RHO, YE, LENGTH, mass_in=3, flavor_out=0)
