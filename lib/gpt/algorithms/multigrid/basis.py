#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


class basis:
    def __init__(self, params, grid):
        self.params = params
        self.preorthog = params["preorthog"]
        self.postorthog = params["postorthog"]
        self.vecstype = params["vecstype"]
        self.numvecs = params["numvecs"]

        self.verbose = True # for now

        self.vecs = [ gpt.vspincolor(grid) for i in range(self.numvecs) ] # TODO: info on type of vectors needs to come from outside
        # cb info is contained within grid object

    def initrandom(self):
        rng = gpt.random("mg_basis_vectors")
        rng.cnormal(self.vecs)

    def generate(self, mat, inv, innercallback=None):
        src, psi = gpt.copy(self.vecs[0]), gpt.copy(self.vecs[0])

        if self.preorthog:
            gpt.orthonormalize(self.vecs)

        for j,v in enumerate(self.vecs):
            if self.vecstype == "test":
                psi[:] = 0
                src @= v
            elif self.vecstype == "null":
                psi @= v
                src[:] = 0
            else:
                raise Exception("Unknown vector type " + self.vecstype)

            inv(mat, src, psi) # can be a solver, sap, mg cycle, or even solver with mg prec
            v @= psi

            if innercallback is not None:
                innercallback(j)

        if self.postorthog:
            gpt.orthonormalize(self.vecs)

    def print_res(self, mat):
        mmv = gpt.copy(self.vecs[0])
        for j,v in enumerate(self.vecs):
            mat(v, mmv)
            gpt.message("j = %d, |M*psi|/|psi| = %g" % (j, gpt.norm2(mmv)/gpt.norm2(v)))
