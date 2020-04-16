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


class intergrid:

    def __init__(self, cgrid, params):
        self.cgrid = cgrid
        self.npasses = params["northo"]

    def recreate(self, basis):
        self.basis = basis  # TODO: ensure deep copy here!
        for i in range(self.npasses):
            gpt.block.orthonormalize(self.cgrid, self.basis)
            gpt.message("Finished aggregate orthonormalization step %d" % (i))

    def fineToCoarse(self, finevec, coarsevec):
        gpt.block.project(coarsevec, finevec, self.basis)

    def coarseToFine(self, coarsevec, finevec):
        gpt.block.promote(coarsevec, finevec, self.basis)
