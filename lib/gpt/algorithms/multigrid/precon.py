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

class precon:

    def __init__(self, params):
        self.params = params
        self.nlevel = params.nlevel
        self.toplevel = None

    def update_setup(self, params):
        self.toplevel.update_setup(params)

    def __call__(self, src, psi):
        self.toplevel(src, psi)
        
    def _append_level(self): # arrange MG levels as doubly-linked list
        newlevel = gpt.algorithms.multigrid.mglevel(self.params)

        # new level is going to be the coarsest in the hierarchy
        newlevel.coarser = None

        # add new level as finest level
        if self.toplevel is None:
            newlevel.nextfiner = None
            newlevel.mylvl = 0
            self.toplevel = newlevel
            return

        # otherwise traverse list until coarsest level
        bottomlvl = self.toplevel
        tmp = 0
        while bottomlvl.next is not None:
            tmp += 1
            bottomlvl = bottomlvl.next

        newlevel.mylvl = tmp
        bottomlvl.nextcoarser = newlevel

        newlevel.nextfiner = bottomlvl
