/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#define prec_double 2
#define prec_single 1

template<int prec, int level>
struct CoarseComplexLvl {
  typedef iScalar<typename CoarseComplexLvl<prec, level-1>::type> type;
};
template<> struct CoarseComplexLvl<prec_double, 0> { typedef vTComplexD type; };
template<> struct CoarseComplexLvl<prec_single, 0> { typedef vTComplexF type; };

template<int nbasis, int prec, int level>
struct FineVectorLvl {
  typedef iVector<typename CoarseComplexLvl<prec, level-1>::type, nbasis> type;
};
template<int nbasis> struct FineVectorLvl<nbasis, prec_double, 0> { typedef vSpinColourVectorD type; };
template<int nbasis> struct FineVectorLvl<nbasis, prec_single, 0> { typedef vSpinColourVectorF type; };

template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_coarsenedmatrix(PyObject* args) {

  constexpr int prec = getPrecision<vCoeff_t>::value; // 2 = double, 1 = single

  auto grid_c = get_pointer<GridCartesian>(args,"grid_c"); // should actually take an 'F_', and an 'U_' grid
  int hermitian = get_int(args,"hermitian");
  int level = get_int(args,"level"); // 0 = fine, increases with coarser levels
  int nbasis = get_int(args,"nbasis");

  // // Tests for the type classes ///////////////////////////////////////////////
  // char* f0 = typename FineVectorLvl<nbasis, prec, 0>::type{}; char* c0 = typename CoarseComplexLvl<prec, 0>::type{};
  // char* f1 = typename FineVectorLvl<nbasis, prec, 1>::type{}; char* c1 = typename CoarseComplexLvl<prec, 1>::type{};
  // char* f2 = typename FineVectorLvl<nbasis, prec, 2>::type{}; char* c2 = typename CoarseComplexLvl<prec, 2>::type{};

#define CASE_FOR_LEVEL(level, nbasis_) \
  case level: \
    typedef typename FineVectorLvl<nbasis_, prec, level>::type FVectorLvl##level; \
    typedef typename CoarseComplexLvl<prec, level>::type       CComplexLvl##level; \
    return new cgpt_fermion_operator<CoarsenedMatrix<FVectorLvl##level, CComplexLvl##level, nbasis_>>( \
      new CoarsenedMatrix<FVectorLvl##level, CComplexLvl##level, nbasis_>(*grid_c, hermitian)); \
    break;

#define BASIS_SIZE(n) \
  if(n == nbasis) { \
    switch(level) { \
      CASE_FOR_LEVEL(0, n); \
      CASE_FOR_LEVEL(1, n); \
      CASE_FOR_LEVEL(2, n); \
      default: ERR("Unknown level %d", level); break; \
    } \
  } else
#include "../basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d", (int)nbasis); }

  // NOTE: With this we should have a default initialized instance of coarsenedmatrix
}

#undef CASE_FOR_LEVEL
#undef prec_double
#undef prec_single
