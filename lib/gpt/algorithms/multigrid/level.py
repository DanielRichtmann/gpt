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

class level:

    # Rationale: Implement mg levels as doubly-linked list
    # Finest is start, coarsest is end
    # -> Cannot add node at start

    def __init__(self, params, finer=None, coarser=None): # Doubly-linked list
        self.params = params
        self.finer = finer
        self.coarser = coarser

        self.checktol = blah

        # each level has a Dirac matrix
        # each level has a smoother
        # each level but the coarsest has an intergrid instance
        # each level but the coarsest has null vectors
        # each level but the finest needs to hold a src (r) and a psi (e) vector

    def iscoarsest(self):
        return self.coarser is None

    def isfinest(self):
        return self.finer is None

    def recreateIntergrid(self):
        if not self.isCoarsest(): # coarset level doesn't have to create an intergrid instance
            pass

    def recreateOperator(self):
        if not self.isfinest(): # finest level doesn't have to create its operator
            pass

    def setupinit(self):
        if not self.iscoarsest():
            self.basis.initrandom()
            self.basis.generate(...)
            self.recreateIntergrid()
            self.recreateOperator()

        if not self.generatealllevels:
            self.projectVectorsDown()

        self.coarser.setupinit()

    def setuprefine(self):
        if not self.iscoarsest():
            for i in range(self.params.nsetupiter):
                self.basis.generate()
                self.recreateIntergrid()
                self.recreateOperator()
            self.coarser.setuprefine()

    def __call__(self, src, psi):
        if self.iscoarsest():
            self.smooth(src, psi)
        else:
            # self.mgcycle(mat, smooth, intergrid, coarser, src, psi)
            self.mgcycle(src, psi)

    def runchecks(self):
        # first check
        for v in self.basis.vectors:
            self.intergrid.fineToCoarse(v, tmp_c[0])  #   R v_i
            self.intergrid.coarseToFine(tmp_c, tmp_f) # P R v_i

            diff_f @= v - tmp_f # v_i - P R v_i
            reldev = (gpt.norm2(diff_f) / gpt.norm2(v))**0.5

            if reldev > self.checktol:
                pass

        # second check
        random(tmp_c[0])
        self.intergrid.coarseToFine(tmp_c[0], tmp_f[0]) #   P v_c
        self.intergrid.fine2Coarse(tmp_f[0], tmp_c[1])  # R P v_c

        diff_c @= tmp_c[0] - tmp_c[1] # v_c - R P v_c
        reldev = (gpt.norm2(diff_c) / gpt.norm2(tmp_c[0]))**0.5

        if reldev > self.checktol:
            pass

        # third check
        random(tmp_c[0])
        self.intergrid.coarseToFine(tmp_c[0], tmp_f[0]) #   P v_c
        self.operator(tmp_f[0], tmp_f[1])               #   D P v_c
        self.intergrid.fineToCoarse(tmp_f[1], tmp_c[1]) # R D P v_c

        self.coarser.operator(tmp_c[0], tmp_c[2]) # D_c v_c

        diff_c @= tmp_c[1] - tmp_c[2]
        reldev = (gpt.norm2(diff_c) / gpt.norm2(tmp_c[1]))**0.5

        if reldev > self.checktol:
            pass

        # fourth check
        random(tmp_f[0])
        self.operator.op(tmp_f[0], tmp_f[1])
        self.operator.adjop(tmp_f[1], tmp_f[2])

        dot = g.innerProduct(tmp_f[0], tmp_f[2])
        dev = numpy.absolute(dot.imag) / numpy.absolute(dot.real)

        # TODO: some logging

        if reldev > self.checktol:
            pass

        # run checks on next coarser level
        self.coarser.runchecks()