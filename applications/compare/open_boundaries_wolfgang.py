#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test open boundary conditions on cls ensembles
#
import gpt as g
import numpy as np
import sys

# setup rng, mute
g.default.set_verbose("random", False)
rng = g.random("open_boundaries")

# parameters
fn = g.default.get(
    "--params",
    "./open_boundaries.M002.txt",
)
params = g.params(fn, verbose=True)

# load configuration
U = g.load(params["config"])
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# default grid
grid = U[0].grid

# create point source, destination
src, dst = g.mspincolor(grid), g.mspincolor(grid)
g.create.point(src, params["src_coor"])

# abbreviations
i = g.algorithms.inverter
p = g.qcd.fermion.preconditioner

for action in ["wilson", "wilson_clover"]:
    # fermion operator
    w = g.qcd.fermion.wilson_clover(U, params["fermion"][action])

    # solvers
    # NOTE: out of the standard krylov solvers, red-black cg is the strictest
    # test since it calls also inv and adj matrices
    slv = w.propagator(
        i.preconditioned(p.eo2_ne(parity=g.odd), i.cg({"eps": 1e-12, "maxiter": 10000}))
    )
    if "--use-fgmres" in sys.argv:
        slv = w.propagator(
            i.fgmres(
                {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
                prec=i.mixed_precision(
                    i.preconditioned(
                        p.eo2(), i.bicgstab({"eps": 1e-3, "maxiter": 1000})
                    ),
                    g.single,
                    g.double,
                ),
            )
        )

    # propagator
    slv(dst, src)

    # two point
    corr = g.slice(g.trace(dst * g.adj(dst)), 3)

    # reference
    corr_ref = params["reference_correlator"][action]

    # deviations per timeslice
    diff = np.array(corr).real - np.array(corr_ref)
    abs_dev = abs(diff)
    denom = [
        corr_ref[t].real if corr_ref[t].real != 0.0 else 1.0
        for t in range(len(corr_ref))
    ]
    rel_dev = np.divide(abs_dev, denom)
    g.message(f"corr: timeslice corr_reference corr abs_dev rel_dev")
    for t in range(len(corr_ref)):
        g.message(
            f"corr: {t:4d} {corr_ref[t].real:20.15e} {corr[t].real:20.15e} {abs_dev[t]:20.15e} {rel_dev[t]:20.15e}"
        )

    # overall error (same as np.linalg.norm(diff)/len(diff))
    eps = 0.0
    for t in range(len(corr_ref)):
        eps += (corr_ref[t].real - corr[t].real) ** 2.0
    eps = eps ** 0.5 / len(corr_ref)
    tol = 1e-12  # g.double.eps?
    g.message(
        f"sqrt(sum_t(diff[t]^2))/len(diff) = {eps} -> check {'passed' if eps <= tol else 'failed'}"
    )
    assert eps <= tol
    g.message("Test successful")
