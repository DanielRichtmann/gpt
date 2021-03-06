#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import gpt
import cgpt
import numpy
import sys


################################################################################
# Merging / Separating along space-time coordinates
################################################################################
def merge(lattices, dimension=-1, N=-1):

    # if only one lattice is given, return immediately
    if type(lattices) != list:
        return lattices

    # number of lattices
    n = len(lattices)
    assert n > 0

    # number of batches
    if N == -1:
        N = n
    batches = n // N
    assert n % N == 0

    # all grids need to be the same
    grid = lattices[0].grid
    assert all([lattices[i].grid.obj == grid.obj for i in range(1, n)])

    # allow negative indexing
    if dimension < 0:
        dimension += grid.nd + 1
        assert dimension >= 0
    else:
        assert dimension <= grid.nd

    # infer checkerboarding of new dimension
    cb = [x.checkerboard() for x in lattices]
    if cb[0] is gpt.none:
        assert all([x is gpt.none for x in cb[1:]])
        cb_mask = 0
    else:
        assert all(
            [
                cb[j * N + i] is cb[j * N + i + 1].inv()
                for i in range(N - 1)
                for j in range(batches)
            ]
        )
        cb_mask = 1

    # otypes must be consistent
    otype = lattices[0].otype
    assert all([lattices[i].otype.__name__ == otype.__name__ for i in range(1, n)])

    # create merged grid
    merged_grid = grid.inserted_dimension(dimension, N, cb_mask=cb_mask)

    # create merged lattices and set checkerboard
    merged_lattices = [gpt.lattice(merged_grid, otype) for i in range(batches)]
    for x in merged_lattices:
        x.checkerboard(cb[0])

    # coordinates of source lattices
    gcoor_zero = lattices[
        0
    ].mview_coordinates()  # return coordinates in internal ordering, speed up access
    gcoor_one = (
        lattices[1].mview_coordinates() if N > 1 and cb_mask == 1 else gcoor_zero
    )
    gcoor = [gcoor_zero, gcoor_one]

    # data transfer
    for i in range(N):
        merged_gcoor = cgpt.coordinates_inserted_dimension(gcoor[i % 2], dimension, [i])
        gpt.poke(
            merged_lattices,
            merged_gcoor,
            gpt.peek([lattices[j * N + i] for j in range(batches)], gcoor[i % 2]),
        )

    # if only one batch, remove list
    if len(merged_lattices) == 1:
        return merged_lattices[0]

    # return
    return merged_lattices


def separate(lattices, dimension=-1):

    # expect list below
    if type(lattices) != list:
        lattices = [lattices]

    # evaluate in case it is an expression
    lattices = [gpt.eval(x) for x in lattices]

    # number of batches to separate
    batches = len(lattices)
    assert batches > 0

    # make sure all have the same grid
    grid = lattices[0].grid
    assert all([lattices[i].grid.obj == grid.obj for i in range(1, batches)])

    # allow negative indexing
    if dimension < 0:
        dimension += grid.nd
        assert dimension >= 0
    else:
        assert dimension < grid.nd

    # number of slices (per batch)
    N = grid.fdimensions[dimension]
    n = N * batches

    # all lattices need to have same checkerboard
    cb = lattices[0].checkerboard()
    assert all([lattices[i].checkerboard() is cb for i in range(1, batches)])

    # all lattices need to have same otype
    otype = lattices[0].otype
    assert all(
        [lattices[i].otype.__name__ == otype.__name__ for i in range(1, batches)]
    )

    # create grid with dimension removed
    separated_grid = grid.removed_dimension(dimension)
    cb_mask = grid.cb.cb_mask[dimension]

    # create separate lattices and set their checkerboard
    separated_lattices = [gpt.lattice(separated_grid, otype) for i in range(n)]
    for i, x in enumerate(separated_lattices):
        j = i % N
        if cb_mask == 0 or j % 2 == 0:
            x.checkerboard(cb)
        else:
            x.checkerboard(cb.inv())

    # construct coordinates
    separated_gcoor_zero = separated_lattices[0].mview_coordinates()
    separated_gcoor_one = (
        separated_lattices[1].mview_coordinates()
        if N > 1 and cb_mask == 1
        else separated_gcoor_zero
    )
    separated_gcoor = [separated_gcoor_zero, separated_gcoor_one]

    # move data
    for i in range(N):
        gcoor = cgpt.coordinates_inserted_dimension(
            separated_gcoor[i % 2], dimension, [i]
        )
        gpt.poke(
            [separated_lattices[j * N + i] for j in range(batches)],
            separated_gcoor[i % 2],
            gpt.peek(lattices, gcoor),
        )

    # return
    return separated_lattices


################################################################################
# Merging / Separating along internal indices
################################################################################
def separate_indices(x, st):
    pos = gpt.coordinates(x)
    cb = x.checkerboard()
    assert st is not None
    result_otype = st[-1]()
    if result_otype is None:
        return x
    ndim = x.otype.shape[st[0]]
    rank = len(st) - 1
    islice = [slice(None, None, None) for i in range(len(x.otype.shape))]
    ivec = [0] * rank
    result = {}
    for i in range(ndim ** rank):
        idx = i
        for j in range(rank):
            c = idx % ndim
            islice[st[j]] = c
            ivec[j] = c
            idx //= ndim
        v = gpt.lattice(x.grid, result_otype)
        v.checkerboard(cb)
        v[pos] = x[(pos,) + tuple(islice)]
        result[tuple(ivec)] = v
    return result


def separate_spin(x):
    return separate_indices(x, x.otype.spintrace)


def separate_color(x):
    return separate_indices(x, x.otype.colortrace)


def merge_indices(dst, src, st):
    pos = gpt.coordinates(dst)
    assert st is not None
    result_otype = st[-1]()
    if result_otype is None:
        dst @= src
        return
    ndim = dst.otype.shape[st[0]]
    rank = len(st) - 1
    islice = [slice(None, None, None) for i in range(len(dst.otype.shape))]
    ivec = [0] * rank
    for i in range(ndim ** rank):
        idx = i
        for j in range(rank):
            c = idx % ndim
            islice[st[j]] = c
            ivec[j] = c
            idx //= ndim
        dst[(pos,) + tuple(islice)] = src[tuple(ivec)][:]


def merge_spin(dst, src):
    merge_indices(dst, src, dst.otype.spintrace)
    return dst


def merge_color(dst, src):
    merge_indices(dst, src, dst.otype.colortrace)
    return dst
