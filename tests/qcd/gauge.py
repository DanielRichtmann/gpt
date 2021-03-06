#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# load configuration
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)
V = rng.lie(g.lattice(U[0]))
U_transformed = g.qcd.gauge.transformed(U, V)

# Test gauge invariance of plaquette
P = g.qcd.gauge.plaquette(U)
P_transformed = g.qcd.gauge.plaquette(U_transformed)
eps = abs(P - P_transformed)
g.message(f"Plaquette before {P} and after {P_transformed} gauge transformation: {eps}")
assert eps < 1e-13

# Test gauge covariance of staple
rho = np.array(
    [[0.0 if i == j else 0.1 for i in range(4)] for j in range(4)], dtype=np.float64
)
C = g.qcd.gauge.smear.staple_sum(U, rho=rho)
C_transformed = g.qcd.gauge.smear.staple_sum(U_transformed, rho=rho)
for mu in range(len(C)):
    q = g.sum(g.trace(C[mu] * g.adj(U[mu]))) / U[0].grid.gsites
    q_transformed = (
        g.sum(g.trace(C_transformed[mu] * g.adj(U_transformed[mu]))) / U[0].grid.gsites
    )

    eps = abs(q - q_transformed)
    g.message(
        f"Staple q[{mu}] before {q} and after {q_transformed} gauge transformation: {eps}"
    )
    assert eps < 1e-14


# Test stout smearing
U_stout = U
P_stout = []
for i in range(3):
    U_stout = g.qcd.gauge.smear.stout(U_stout, rho=0.1)

    for mu in range(len(U_stout)):
        I = g.identity(U_stout[mu])
        eps2 = g.norm2(U_stout[mu] * g.adj(U_stout[mu]) - I) / g.norm2(I)
        g.message(f"Unitarity check of stout-smeared links: mu = {mu}, eps2 = {eps2}")

    P_stout.append(g.qcd.gauge.plaquette(U_stout))

g.message(f"Stout smeared plaquettes {P_stout}")
assert sorted(P_stout) == P_stout  # make sure plaquettes go towards one

# for given gauge configuration, cross-check against previous Grid code
# this establishes the randomized check value used below
# U = g.load("/hpcgpfs01/work/clehner/configs/24I_0p005/ckpoint_lat.IEEE64BIG.5000")
# P = [g.qcd.gauge.plaquette(U),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1))]
# P_comp = [0.588074,0.742136,0.820262]
# for i in range(3):
#    assert abs(P[i] - P_comp[i]) < 1e-5
# g.message(f"Plaquette fingerprint {P} and reference {P_comp}")

P = [
    g.qcd.gauge.plaquette(U),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1)),
]
P_comp = [0.7986848674527128, 0.9132213221481771, 0.9739960794712376]
g.message(f"Plaquette fingerprint {P} and reference {P_comp}")
for i in range(3):
    assert abs(P[i] - P_comp[i]) < 1e-12
