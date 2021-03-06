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
import cgpt, gpt, numpy


def cshift(first, second, third, fourth=None):

    if (
        type(first) == gpt.lattice
        and type(second) == gpt.lattice
        and fourth is not None
    ):
        t = first
        l = gpt.eval(second)
        d = third
        o = fourth
    else:
        l = gpt.eval(first)
        d = second
        o = third
        t = gpt.lattice(l)

    for i in t.otype.v_idx:
        cgpt.cshift(t.v_obj[i], l.v_obj[i], d, o)
    return t


def copy(first, second=None):

    if type(first) == gpt.lattice:
        if second is not None:
            t = first
            l = second
        else:
            l = first
            t = gpt.lattice(l)
        for i in t.otype.v_idx:
            cgpt.copy(t.v_obj[i], l.v_obj[i])
        return t

    else:
        assert 0


def convert(first, second):
    if second in [gpt.single, gpt.double]:

        # if first is a list, distribute
        if type(first) == list:
            return [convert(x, second) for x in first]

        # if first is no list, evaluate
        src = gpt.eval(first)
        dst_grid = src.grid.converted(second)
        return convert(gpt.lattice(dst_grid, src.otype), src)

    elif type(first) == list:

        assert len(first) == len(second)
        for i in range(len(first)):
            convert(first[i], second[i])
        return first

    elif type(first) == gpt.lattice:

        # second may be expression
        second = gpt.eval(second)

        # now second is lattice
        assert len(first.otype.v_idx) == len(second.otype.v_idx)
        for i in first.otype.v_idx:
            cgpt.convert(first.v_obj[i], second.v_obj[i])

        # set checkerboard
        first.checkerboard(second.checkerboard())
        return first

    else:
        assert 0


def rank_inner_product(a, b, use_accelerator=True):
    return_list = (type(a) == list) or (type(b) == list)
    a = gpt.util.to_list(a)
    b = gpt.util.to_list(b)
    if type(a[0]) == gpt.tensor and type(b[0]) == gpt.tensor:
        res = numpy.array(
            [[gpt.adj(x) * y for y in b] for x in a], dtype=numpy.complex128
        )
    else:
        a = [gpt.eval(x) for x in a]
        b = [gpt.eval(x) for x in b]
        otype = a[0].otype
        assert len(otype.v_idx) == len(b[0].otype.v_idx)
        res = cgpt.lattice_rank_inner_product(a, b, use_accelerator)
    if return_list:
        return res
    return gpt.util.to_num(res[0, 0])


def inner_product(a, b):
    grid = gpt.util.to_list(a)[0].grid
    return grid.globalsum(rank_inner_product(a, b))


def norm2(l):
    if type(l) == gpt.tensor:
        return l.norm2()
    l = gpt.eval(l)  # otherwise it gets evaluated twice below
    return inner_product(l, l).real


def inner_product_norm2(a, b):
    if type(a) == gpt.tensor and type(b) == gpt.tensor:
        return gpt.adj(a) * b, a.norm2()
    a = gpt.eval(a)
    b = gpt.eval(b)
    assert len(a.otype.v_idx) == len(b.otype.v_idx)
    r = [
        cgpt.lattice_inner_product_norm2(a.v_obj[i], b.v_obj[i]) for i in a.otype.v_idx
    ]
    return (
        sum([x[0] for x in r]),
        sum([x[1] for x in r]),
    )  # todo, make local version of this too


def axpy_norm2(d, a, x, y):
    x = gpt.eval(x)
    y = gpt.eval(y)
    assert len(y.otype.v_idx) == len(x.otype.v_idx)
    assert len(d.otype.v_idx) == len(x.otype.v_idx)
    return sum(
        [
            cgpt.lattice_axpy_norm2(d.v_obj[i], a, x.v_obj[i], y.v_obj[i])
            for i in x.otype.v_idx
        ]
    )


def axpy(d, a, x, y):
    x = gpt.eval(x)
    y = gpt.eval(y)
    assert len(y.otype.v_idx) == len(x.otype.v_idx)
    assert len(d.otype.v_idx) == len(x.otype.v_idx)
    for i in x.otype.v_idx:
        cgpt.lattice_axpy(d.v_obj[i], a, x.v_obj[i], y.v_obj[i])


def slice(x, dim):
    x = gpt.eval(x)
    r = sum([numpy.array(cgpt.lattice_slice(o, dim)) for o in x.v_obj])
    return [gpt.util.value_to_tensor(v, x.otype) for v in r]


def identity(src):
    eye = gpt.lattice(src)
    eye[:] = src.otype.identity()
    return eye
