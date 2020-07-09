#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann
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
import gpt as g


class mr:

    # Y. Saad calls it MR, states mat must be positive definite
    # SciPy, Wikipedia call it MINRES, state mat must be symmetric

    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.relax = params["relax"]
        self.history = None

    def __call__(self, mat):
        def inv(psi, src):
            self.history = []
            verbose = g.default.is_verbose("mr")

            t = g.timer()
            t.start("total")
            t.start("setup")

            r, mmr = g.copy(src), g.copy(src)

            mat(mmr, psi)
            r @= src - mmr

            ssq = g.norm2(src)
            rsq = self.eps ** 2.0 * ssq

            t.stop("setup")

            for k in range(self.maxiter):
                t.start("mat")
                mat(mmr, r)
                t.stop("mat")

                t.start("inner")
                ip, mmr2 = g.innerProductNorm2(mmr, r)
                t.stop("inner")

                if mmr2 == 0.0:
                    continue

                t.start("linearcomb")
                alpha = ip.real / mmr2 * self.relax
                psi += alpha * r
                t.stop("linearcomb")

                t.start("axpy_norm")
                r2 = g.axpy_norm2(r, -alpha, mmr, r)
                t.stop("axpy_norm")

                self.history.append(r2)

                if verbose:
                    g.message("mr: res^2[ %d ] = %g, target = %g" % (k, r2, rsq))

                if r2 <= rsq:
                    if verbose:
                        t.stop("total")
                        g.message(
                            "mr: converged in %d iterations, took %g s"
                            % (k + 1, t.dt["total"])
                        )
                        t.print("mr")
                    break

        otype = None
        grid = None
        if type(mat) == g.matrix_operator:
            otype = mat.otype
            grid = mat.grid

        return g.matrix_operator(
            mat=inv, inv_mat=mat, otype=otype, zero=(False, False), grid=grid
        )
