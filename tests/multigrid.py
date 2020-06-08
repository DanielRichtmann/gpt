#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Develop multigrid
#
import gpt as g
import numpy as np
import sys
import time
import os.path

# load configuration
homedir = os.path.expanduser("~")
U = g.load(homedir + "/configs/openqcd/test_16x8_pbcn6")
# U = g.load(homedir + "/configs/nersc/test_16x16_pbcn0")
# U = g.load("/hpcgpfs01/work/clehner/configs/32IDfine/ckpoint_lat.200") # TODO: add parallel RNG so we can do tests from random gauge configs

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# coarse grid
blocksize=[4,4,4,4]
grid_coarse=g.block.grid(grid,blocksize)

# clover fermions
w=g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.13565,
    "csw_r" : 2.0171 / 2., # for now test with very heavy quark
    "csw_t" : 2.0171 / 2.,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# mg basis vectors
mgb = g.algorithms.multigrid.basis({
    "preorthog": False,
    "postorthog": True,
    "vecstype": "null",
    # "vecstype": "test",
    "numvecs": 20
}, grid)

# start from random vectors
mgb.initrandom()
mgb.print_res(w.M)

# mr as initial setup solver (for now)
mr = g.algorithms.iterative.mr({
    "eps": 1e-1,
    "maxiter": 4,
    "relax": 1.0
})

# find approx near-null vectors with smoother
mgb.generate(w.M, mr)
mgb.print_res(w.M)

# create intergrid operator from near-null vectors
mgi = g.algorithms.multigrid.intergrid(grid_coarse, {
    "northo": 2
})
mgi.recreate(mgb.vecs)

# temporary vectors
rng = g.random("foo_bar_buz_uiae_asdf")
tmpc1 = g.vcomplex(grid_coarse, 20)
tmpc2 = g.vcomplex(grid_coarse, 20)
tmpf1 = g.lattice(mgb.vecs[0])
tmpf2 = g.lattice(mgb.vecs[0])

# randomize vectors
rng.cnormal(tmpc1)
rng.cnormal(tmpc2)
rng.cnormal(tmpf1)
rng.cnormal(tmpf2)

# test intergrid operators (1 - PR)v_f == 0 (NOT working currently)
mgi.fineToCoarse(tmpf1, tmpc1)
mgi.coarseToFine(tmpc1, tmpf2)
g.message(
    "v_f = %g, Rv_f = %g, PRv_f = %g, (1 - PR)v_f = %g" %
    (g.norm2(tmpf1), g.norm2(tmpc1), g.norm2(tmpf2), g.norm2(tmpf1 - tmpf2)))

# re-randomize vectors
rng.cnormal(tmpc1)
rng.cnormal(tmpc2)
rng.cnormal(tmpf1)
rng.cnormal(tmpf2)

# test intergrid operators (1 - RP)v_c == 0 (NOT working currently)
mgi.coarseToFine(tmpc1, tmpf1)
mgi.fineToCoarse(tmpf1, tmpc2)
g.message(
    "v_c = %g, Pv_c = %g, RPv_c = %g, (1 - RP)v_c = %g" %
    (g.norm2(tmpc1), g.norm2(tmpf1), g.norm2(tmpc2), g.norm2(tmpc1 - tmpc2)))
