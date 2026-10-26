"""Random-point check of the solar ``method="quadratic"`` propagation on the SF-III/GS98
profile: from a single production radius against thin constant-density layers (640k,
checked against 320k) through the same profile, for 3 flavors with and without large
NSI, and 3+1 over theta_i4 in [1e-4, 0.6] and |Delta m^2_41| in [1e-9, 10] eV^2, for
0.1-20 MeV, both orderings, neutrinos and antineutrinos.

Run from the repo root (a few minutes on one CPU core):
    JAX_PLATFORMS=cpu python benchmarks/check_solar_quadratic.py [n_points]
"""
import dataclasses
import os
import sys
import warnings

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mango import NSI, Sterile3plus1, nufit_no, solar, constants as C
from mango.layers import propagate_layers

warnings.simplefilter("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(1)
prof = solar.load_sf3(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data",
                                   "struct+nu_SF3_GS98.dat"))
r = np.asarray(prof.r_over_rsun)
base = nufit_no()


def log_uniform(a, b):
    return 10 ** rng.uniform(np.log10(a), np.log10(b))


def sterile(th14, th24, th34, dm41, dm31):
    return Sterile3plus1(theta12=base.theta12, theta13=base.theta13, theta23=base.theta23,
                         theta14=jnp.asarray(th14), theta24=jnp.asarray(th24), theta34=jnp.asarray(th34),
                         delta13=base.deltacp, delta24=jnp.asarray(rng.uniform(0, 2 * np.pi)), dm21=base.dm21,
                         dm31=jnp.asarray(dm31), dm41=jnp.asarray(dm41), delta14=jnp.asarray(rng.uniform(0, 2 * np.pi)))


def layers(p, e, r0, anti, nsi, n):
    edges = np.linspace(r0, r[-1], n + 1)
    x = 0.5 * (edges[1:] + edges[:-1])
    s = propagate_layers(p.pmns(), p.msquared(), e * C.GEV_TO_EV,
                         C.matter_potential_eV(jnp.asarray(np.interp(x, r, prof.rho_ye)), 1.0),
                         jnp.asarray(np.diff(edges) * prof.R_sun_km * C.KM_TO_INV_EV), anti=anti,
                         nsi=None if nsi is None else solar._nsi_matrix(nsi, 3),
                         v_nc=0.5 * C.matter_potential_eV(jnp.asarray(np.interp(x, r, prof.rho_yn)), 1.0),
                         n_active=3, backend="eigh")
    u = np.asarray(p.pmns())
    return np.abs((u if anti else np.conj(u)).T @ np.asarray(s)[:, 0]) ** 2


errors, ref_conv = [], []
for _ in range(N):
    io, anti = rng.random() < 0.3, bool(rng.random() < 0.3)
    dm31 = float(base.dm21) - float(base.dm31) if io else float(base.dm31)
    e, r0 = log_uniform(1e-4, 2e-2), r[np.argmin(np.abs(r - rng.uniform(0.0, 0.35)))]
    nsi = None
    if rng.random() < 0.35:
        p = dataclasses.replace(base, dm31=dm31)
        if rng.random() < 0.6:
            def off():
                return rng.uniform(-0.5, 0.5) + 1j * rng.uniform(-0.5, 0.5)
            nsi = NSI(eps_ee=rng.uniform(-2, 2), eps_mumu=rng.uniform(-1, 1), eps_tautau=rng.uniform(-1, 1),
                      eps_emu=off(), eps_etau=off(), eps_mutau=off())
    else:
        def angle():
            return log_uniform(1e-4, 0.6) if rng.random() < 0.6 else 0.0
        p = sterile(log_uniform(1e-4, 0.6), angle(), angle(),
                    log_uniform(1e-9, 10.0) * (-1 if rng.random() < 0.15 else 1), dm31)
    got = np.asarray(solar.production_mixing(p, e, prof, (r == r0).astype(float), anti=anti, nsi=nsi,
                                             method="quadratic"))
    ref, coarse = layers(p, e, r0, anti, nsi, 640000), layers(p, e, r0, anti, nsi, 320000)
    errors.append(np.abs(got - ref).max())
    ref_conv.append(np.abs(ref - coarse).max())
errors, ref_conv = np.array(errors), np.array(ref_conv)
print(f"quadratic vs layers, {N} points: max {errors.max():.1e}, median {np.median(errors):.1e}, "
      f"above 1e-5: {np.sum(errors > 1e-5)}; reference converged to {np.median(ref_conv):.0e} (median)")
