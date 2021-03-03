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
mg = i.multi_grid

# for action in ["wilson", "wilson_clover"]:
# for action in ["wilson"]:
for action in ["wilson_clover"]:
    # fermion operator
    w = g.qcd.fermion.wilson_clover(U, params["fermion"][action])

    # solvers
    # NOTE: out of the standard krylov solvers, red-black cg is the strictest
    # test since it calls also inv and adj matrices
    slv = w.propagator(
        i.preconditioned(p.eo2_ne(parity=g.odd), i.cg({"eps": 1e-12, "maxiter": 10000}))
    )
    if "--use-split-cg" in sys.argv:
        slv = w.propagator(
            i.preconditioned(
                p.eo2_ne(parity=g.odd),
                i.split(
                    i.cg({"eps": 1e-12, "maxiter": 10000}),
                    mpi_split=g.default.get_ivec("--mpi_split", None, 4),
                ),
            )
        )
    elif "--use-fgmres" in sys.argv:
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
    if "--use-split-fgmres" in sys.argv:
        slv = w.propagator(
            i.split(
                i.fgmres(
                    {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
                    prec=i.mixed_precision(
                        i.preconditioned(
                            p.eo2(), i.bicgstab({"eps": 1e-3, "maxiter": 1000})
                        ),
                        g.single,
                        g.double,
                    ),
                ),
                mpi_split=g.default.get_ivec("--mpi_split", None, 4),
            )
        )
    elif "--use-multigrid" in sys.argv:
        w_s = w.converted(g.single)
        mg_setup = mg.setup(
            w_s,
            # lists control levels separately, scalars are broadcasted
            {
                # mapping between fine and coarser lattices, control num levels
                "block_size": [[3, 3, 3, 2], [2, 2, 2, 2]]
                if "--3-lvl" in sys.argv
                else [[6, 3, 3, 4]],
                "n_block_ortho": 2,  # number of Gram Schmidt passes for block ortho, 1 - 2 is fine
                "check_block_ortho": True,  # perform orthogonality check
                "n_basis": [48, 60]
                if "--3-lvl" in sys.argv
                else [48],  # number of null vectors per level * 2
                "make_hermitian": False,  # not relevant for wilson
                "save_links": True,  # set to True, speeds up setup by saving coarsening directions
                "vector_type": "null",  # leave to null for now ("null" or "test" vectors)
                "n_pre_ortho": 0,  # can be 1, but doesn't have to
                "n_post_ortho": 1,  # should be 1
                "solver": i.preconditioned(
                    p.eo2_ne(parity=g.odd),
                    i.cg({"eps": 1e-6, "maxiter": 1000, "checkres": False}),
                    # p.eo2(parity=g.odd),
                    # i.fgmres({"eps": 1e-6, "maxiter": 1000, "checkres": False}),
                ),  # solver to use for finding null vectors, tolerance ~ 1e-5
                "distribution": rng.cnormal,  # distribution to use for initial null vectors
            },
        )
        mg_prec = mg.inverter(
            mg_setup,
            # again, lists control levels separately
            {
                # solver to use on the coarsest level
                "coarsest_solver": i.preconditioned(
                    # p.eo2_ne(parity=g.odd),
                    # i.cg(
                    p.eo2(parity=g.odd),
                    i.fgmres(
                        {
                            "eps": 0.05,  # 0.25 - 0.05 is fine
                            "maxiter": 1000,  # 8 - 16
                            "restartlen": 20,  # >= maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                # solver to smooth on the levels
                "smooth_solver": i.preconditioned(
                    p.eo2(parity=g.odd),
                    i.fgmres(
                        {
                            "eps": 1e-1,
                            "maxiter": 16,  # 8 - 16
                            "restartlen": 16,  # => maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                # "smooth_solver": i.defect_correcting(
                #     p.sap_cycle(
                #         i.preconditioned(
                #             p.eo2(parity=g.odd),
                #             i.mr({"eps": 1e-16, "maxiter": 4, "relax": 1}),
                #         ),
                #         bs=[4, 4, 4, 4],
                #     ),
                #     {"eps": 1e-1, "maxiter": 4},
                # ),
                # solver used in k-cycle
                "wrapper_solver": None,  # performs v-cycle
                # "wrapper_solver": i.fgmres(  # performs k-cycle
                #     {"eps": 0.1, "maxiter": 16, "restartlen": 8, "checkres": False}
                # ),
            },
        )
        # old version I had before
        slv = w.propagator(
            i.fgmres(
                # {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
                {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
                prec=i.mixed_precision(
                    mg_prec,
                    g.single,
                    g.double,
                ),
            )
        )
    elif "--use-kate-multigrid" in sys.argv:
        w_s = w.converted(g.single)
        mg_setup = mg.setup(
            w_s,
            # lists control levels separately, scalars are broadcasted
            {
                # mapping between fine and coarser lattices, control num levels
                "block_size": [[3, 3, 3, 2], [2, 2, 2, 2]]
                if "--3-lvl" in sys.argv
                else [[6, 3, 3, 4]],
                "n_block_ortho": 2,  # number of Gram Schmidt passes for block ortho, 1 - 2 is fine
                "check_block_ortho": True,  # perform orthogonality check
                "n_basis": [48, 60]
                if "--3-lvl" in sys.argv
                else [48],  # number of null vectors per level * 2
                "make_hermitian": False,  # not relevant for wilson
                "save_links": True,  # set to True, speeds up setup by saving coarsening directions
                "vector_type": "null",  # leave to null for now ("null" or "test" vectors)
                "n_pre_ortho": 0,  # can be 1, but doesn't have to
                "n_post_ortho": 1,  # should be 1
                "solver": i.preconditioned(
                    p.eo2_ne(parity=g.odd),
                    i.cg({"eps": 1e-6, "maxiter": 1000, "checkres": False}),
                    # p.eo2(parity=g.odd),
                    # i.fgmres({"eps": 1e-6, "maxiter": 1000, "checkres": False}),
                ),  # solver to use for finding null vectors, tolerance ~ 1e-5
                "distribution": rng.cnormal,  # distribution to use for initial null vectors
            },
        )
        mg_prec = mg.inverter(
            mg_setup,
            # again, lists control levels separately
            {
                # terminology difference:
                # - our coarsest solver is just the smoother on the coarsest level
                # - our wrapper solver is her coarse solver (she applies that to all but the finest level, i.e., her coarsest grid is the the lowest coarse_solver preconditioned by a smoother)
                # in her terminology, our coarsest solver is just the smoother on the coarsest level, and the wrapper solver on every level is the coarse solver
                "coarsest_solver": i.preconditioned(
                    p.eo2(parity=g.odd),
                    i.mr(
                        {
                            "eps": 0.25,  # 0.25 - 0.05 is fine
                            "maxiter": 8,  # 8 - 16
                            "restartlen": 8,  # >= maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                # solver to smooth on the levels
                "smooth_solver": i.preconditioned(
                    p.eo2(parity=g.odd),
                    i.mr(
                        {
                            "eps": 0.25,
                            "maxiter": 8,  # 8 - 16
                            "restartlen": 8,  # => maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                "wrapper_solver": i.fgcr(
                    {"eps": 0000.1, "maxiter": 12, "restartlen": 12, "checkres": False}
                    # {"eps": 0.25, "maxiter": 12, "restartlen": 12, "checkres": False}
                ),
            },
        )
        # old version I had before
        slv = w.propagator(
            i.fgmres(
                {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
                prec=i.mixed_precision(
                    mg_prec,
                    g.single,
                    g.double,
                ),
            )
        )
    elif "--use-new-multigrid" in sys.argv:
        w_s = w.converted(g.single)
        mg_setup = mg.setup(
            w_s,
            # lists control levels separately, scalars are broadcasted
            {
                # mapping between fine and coarser lattices, control num levels
                "block_size": [[3, 3, 3, 2], [2, 2, 2, 2]]
                if "--3-lvl" in sys.argv
                else [[6, 3, 3, 4]],
                "n_block_ortho": 2,  # number of Gram Schmidt passes for block ortho, 1 - 2 is fine
                "check_block_ortho": True,  # perform orthogonality check
                "n_basis": [48, 60]
                if "--3-lvl" in sys.argv
                else [48],  # number of null vectors per level * 2
                "make_hermitian": False,  # not relevant for wilson
                "save_links": True,  # set to True, speeds up setup by saving coarsening directions
                "vector_type": "null",  # leave to null for now ("null" or "test" vectors)
                "n_pre_ortho": 0,  # can be 1, but doesn't have to
                "n_post_ortho": 1,  # should be 1
                "solver": i.preconditioned(
                    p.eo2_ne(parity=g.odd),
                    i.cg({"eps": 5e-6, "maxiter": 500, "checkres": False}),
                    # p.eo2(parity=g.odd),
                    # i.fgmres({"eps": 1e-6, "maxiter": 1000, "checkres": False}),
                ),  # solver to use for finding null vectors, tolerance ~ 1e-5
                "distribution": rng.cnormal,  # distribution to use for initial null vectors
            },
        )
        mg_prec = mg.inverter(
            mg_setup,
            {
                # solver to use on the coarsest level
                "coarsest_solver": i.preconditioned(
                    p.eo2(parity=g.odd),
                    i.mr(
                        {
                            "eps": 0.25,  # 0.25 - 0.05 is fine
                            "maxiter": 8,  # 8 - 16
                            "restartlen": 8,  # >= maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                # solver to smooth on the levels
                "smooth_solver": i.preconditioned(
                    p.eo2(parity=g.odd),
                    i.mr(
                        {
                            "eps": 0.25,
                            "maxiter": 8,  # 8 - 16
                            "restartlen": 8,  # => maxiter
                            "checkres": False,  # False is ok here
                        }
                    ),
                ),
                # solver used in k-cycle
                # "wrapper_solver": None,  # performs v-cycle
                "wrapper_solver": i.fgmres(  # performs k-cycle
                    {"eps": 0.1, "maxiter": 12, "restartlen": 12, "checkres": False}
                ),
            },
        )
        # new version from Christoph
        slv = w.propagator(
            i.defect_correcting(
                i.mixed_precision(
                    i.fgmres(
                        {"eps": 1e-6, "maxiter": 1000, "restartlen": 20},
                        prec=mg_prec,
                    ),
                    g.single,
                    g.double,
                ),
                eps=1e-12,
                maxiter=15,
            )
        )
        # slv = w.propagator(
        #     i.split(
        #         i.fgmres(
        #             {"eps": 1e-12, "maxiter": 1000, "restartlen": 20},
        #             prec=i.mixed_precision(
        #                 mg_prec,
        #                 g.single,
        #                 g.double,
        #             ),
        #         ),
        #         mpi_split=g.default.get_ivec("--mpi_split", None, 4),
        #     )
        # )

    if "--single-solve" in sys.argv:
        # create point source, destination (only need 1 vector here)
        src, dst = g.vspincolor(grid), g.vspincolor(grid)
        src[:] = 0
        src[[params["src_coor"]]] = 1.0

        t0 = g.time()
        slv(dst, src)
        dt = g.time() - t0
        g.message(f"Inversion took {dt} seconds")

        # report TODO remove
        if "--use-multigrid" in sys.argv or "--use-kate-multigrid" in sys.argv:
            [g.message(t) for t in mg_setup.t if t]
            [g.message(t) for t in mg_prec.t if t]

        sys.exit(0)

    # propagator
    t0 = g.time()
    slv(dst, src)
    dt = g.time() - t0
    g.message(f"Propagator calculation took {dt} seconds -> {dt/12} seconds per solve")

    # report TODO remove
    if "--use-multigrid" in sys.argv:
        [g.message(t) for t in mg_setup.t if t]
        [g.message(t) for t in mg_prec.t if t]

    # two point
    corr = g.slice(g.trace(dst * g.adj(dst)), 3)

    if "reference_correlator" not in params:
        g.message("Aborting since no reference correlator is present in the input file")
        sys.exit(0)

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
