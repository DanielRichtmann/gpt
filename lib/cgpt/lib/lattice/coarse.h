/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

template<typename CMat>
void cgpt_invert_coarse_link(std::vector<cgpt_Lattice_base*>& _link_inv,
                             std::vector<cgpt_Lattice_base*>& _link,
                             long                             n_virtual) {

  PVector<Lattice<CMat>> link_inv;
  PVector<Lattice<CMat>> link;
  cgpt_basis_fill(link_inv, _link_inv);
  cgpt_basis_fill(link, _link);

  invertCoarseLink(link_inv, link, n_virtual);
}
