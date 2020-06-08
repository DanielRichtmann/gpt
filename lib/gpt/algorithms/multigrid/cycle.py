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
from time import time

class cycle:

    def __init__(self, params):
        self.params = params
        self.presmooth = params["presmooth"]
        self.postsmooth = params["postsmooth"]
        self.strategy = params["strategy"]
        assert(self.presmooth == True or self.postsmooth == True) # need some kind of smoothing

        # TODO: distinguish v, w, and k-cycles

    def __call__(self, mat, smooth, intergrid, nextcoarser, src, psi): # psi is only written to, the actual value it holds when passed in is not used
        verbose=gpt.default.is_verbose("mgcycle")

        r = gpt.copy(src)

        t0 = time()
        if self.presmooth:
            tmp, mmtmp = gpt.copy(src), gpt.copy(src)
            smooth(src, tmp) # TODO: need to ensure that tmp is zero before this!
            mat(tmp, mmtmp)
            r2 = gpt.axpy_norm(r, -1., mmtmp, src)
        else:
            r @= src
        t1 = time()

        t2 = time()
        intergrid.fineToCoarse(r, nextcoarser.r)
        t3 = time()

        t4 = time()
        if self.strategy == "K":
            pass
            # TODO: call to solver on next level, preconditioned by next mg level
        else:
            nextcoarser(nextcoarser.r, nextcoarser.e) # call the next level mg cycle directly, i.e., v-cycle
        t5 = time()

        t6 = time()
        intergrid.coarseToFine(nextcoarser.e, psi) # does =, not +=
        t7 = time()

        t6 = time()
        if self.postsmooth:
            smooth(src, psi) # use psi as starting guess
        t7 = time()

        if verbose:
            gpt.message(
                "Timing[s]: Presmooth = %g, Project = %g, CoarseSolve = %g, Promote = %g, Postsmooth = %g"
                % (t1 - t0, t3 - t2, t5 - t4, t7 - t6))
