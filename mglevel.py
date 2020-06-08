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
import gpt as g
from time import time

class mglevel:

    def __init__(self, params):
        self.params = params
        self.presmooth = params["presmooth"]
        self.postsmooth = params["postsmooth"]
        self.strategy = params["strategy"]

    def __call__(self, matsmooth, matcoarse, src, psi):
        verbose=g.default.is_verbose("mgcycle")

        t = {}
        if self.presmooth:
            t['presmooth'] = -time()
            matsmooth(src, tmp)
            t['presmooth'] += time()

        t['project'] = -time()
        pass
        t['project'] += time()

        t['coarse'] = -time()
        pass
        t['coarse'] += time()

        t['promote'] = -time()
        pass
        t['promote'] += time()

        if self.postsmooth:
            t['postsmooth'] = -time()
            pass
            t['postsmooth'] += time()

        if verbose:
            # TODO print that out sorted
            pass


class mgprec:

    # NOTE: Implement this is a doubly linked list of mglevel instances
    # The idea behind this is that it gives us more flexiblity compared
    # to a standard implementation

    def __init__(self, params):
        self.params = params
        self.nlevel = params["nlevel"]
        self.smoother = params["smoother"]
        self.coarse = params["coarse"]

    def setup(self):
        pass

    def __call__(self, src, psi):
        pass
