/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#include "lib.h"
#include "eval/mul_vlat_vlat.h"
#include "eval/matmul_vlat.h"

struct _eval_factor_ {
  enum { LATTICE, ARRAY, GAMMA } type;
  int unary;
  std::vector<cgpt_Lattice_base*> vlattice;
  PyArrayObject* array;

  Gamma::Algebra gamma;
  std::vector<std::string> v_otype;

  void release() {
    if (type == LATTICE) {
      for (auto&l : vlattice)
	delete l;
    } else if (type == ARRAY) {
      Py_DECREF(array);
    }
  }
};

struct _eval_term_ {
  ComplexD coefficient;
  std::vector<_eval_factor_> factors;
};

void eval_convert_factors(PyObject* _list, std::vector<_eval_term_>& terms, int idx) {
  ASSERT(PyList_Check(_list));
  int n = (int)PyList_Size(_list);

  terms.resize(n);
  for (int i=0;i<n;i++) {
    auto& term = terms[i];
    PyObject* tp = PyList_GetItem(_list,i);
    ASSERT(PyTuple_Check(tp) && PyTuple_Size(tp) == 2);

    cgpt_convert(PyTuple_GetItem(tp,0),term.coefficient);

    PyObject* ll = PyTuple_GetItem(tp,1);
    ASSERT(PyList_Check(ll));
    int m = (int)PyList_Size(ll);

    term.factors.resize(m);
    for (int j=0;j<m;j++) {
      auto& factor = term.factors[j];
      PyObject* jj = PyList_GetItem(ll,j);
      ASSERT(PyTuple_Check(jj) && PyTuple_Size(jj) == 2);

      factor.unary = PyLong_AsLong(PyTuple_GetItem(jj,0));
      PyObject* f = PyTuple_GetItem(jj,1);
      if (PyList_Check(f)) {
	ASSERT(idx >= 0 && idx < PyList_Size(f));
	f = PyList_GetItem(f, idx);
      } else {
	ASSERT(idx == 0);
      }
      if (PyObject_HasAttrString(f,"v_obj")) {
	PyObject* v_obj = PyObject_GetAttrString(f,"v_obj");
	ASSERT(v_obj);
	cgpt_convert(v_obj, factor.vlattice);
	factor.type = _eval_factor_::LATTICE;
      } else if (PyObject_HasAttrString(f,"array")) {
	factor.array = (PyArrayObject*)PyObject_GetAttrString(f,"array");
	PyObject* otype = PyObject_GetAttrString(f,"otype");
	ASSERT(otype);
	PyObject* v_otype = PyObject_GetAttrString(otype,"v_otype");
	ASSERT(v_otype);
	cgpt_convert(v_otype,factor.v_otype);
	factor.type = _eval_factor_::ARRAY;
      } else if (PyObject_HasAttrString(f,"gamma")) {
	int gamma = (int)PyLong_AsLong(PyObject_GetAttrString(f,"gamma"));
	ASSERT(gamma >= 0 && gamma < gamma_algebra_map_max);
	factor.gamma = gamma_algebra_map[gamma];
	factor.type = _eval_factor_::GAMMA;
      } else {
	ASSERT(0);
      }
    }
  }
}

_eval_factor_ eval_mul_factor(_eval_factor_ lhs, _eval_factor_ rhs, int unary) {

  _eval_factor_ dst;
  dst.unary = 0;

  if (lhs.type == _eval_factor_::LATTICE) {
    if (rhs.type == _eval_factor_::LATTICE) {
      dst.type = _eval_factor_::LATTICE;
      eval_mul_vlat_vlat(dst.vlattice, lhs.vlattice, lhs.unary, rhs.vlattice, rhs.unary, unary);
    } else if (rhs.type == _eval_factor_::ARRAY) {
      dst.type = _eval_factor_::LATTICE;
      eval_matmul_vlat(dst.vlattice, lhs.vlattice, lhs.unary, rhs.array, rhs.v_otype, rhs.unary, unary, false);
    } else if (rhs.type == _eval_factor_::GAMMA) {
      ASSERT(rhs.unary == 0);
      dst.type = _eval_factor_::LATTICE;
      dst.vlattice.resize(1);
      ASSERT(lhs.vlattice.size() == 1);
      dst.vlattice[0] = lhs.vlattice[0]->gammamul( 0, false, rhs.gamma, lhs.unary, unary, false);
    } else {
      ASSERT(0);
    }
  } else if (lhs.type == _eval_factor_::ARRAY) {
    ASSERT(lhs.unary == 0);
    if (rhs.type == _eval_factor_::LATTICE) {
      dst.type = _eval_factor_::LATTICE;
      eval_matmul_vlat(dst.vlattice, rhs.vlattice, rhs.unary, lhs.array, lhs.v_otype, lhs.unary, unary, true);
    } else {
      ASSERT(0);
    }
  } else if (lhs.type == _eval_factor_::GAMMA) {
    ASSERT(lhs.unary == 0);
    if (rhs.type == _eval_factor_::LATTICE) {
      dst.type = _eval_factor_::LATTICE;
      dst.vlattice.resize(1);
      ASSERT(rhs.vlattice.size() == 1);
      dst.vlattice[0] = rhs.vlattice[0]->gammamul( 0, false, lhs.gamma, rhs.unary, unary, true);
    } else {
      ASSERT(0);
    }
  } else {
    ASSERT(0);
  }

  return dst;
}

std::vector<cgpt_Lattice_base*> eval_term(std::vector<_eval_factor_>& factors, int term_unary) {
  ASSERT(factors.size() > 1);

  //f[im2] = eval_mul_factor(f[im2],f[im1]);
  //f[im3] = eval_mul_factor(f[im3],f[im2]);
  //f[im2].release();
  //f[im4] = eval_mul_factor(f[im4],f[im3]);
  //f[im3].release();

  for (size_t i = factors.size() - 1; i > 0; i--) {
    auto& im1 = factors[i];
    auto& im2 = factors[i-1];
    im2 = eval_mul_factor(im2,im1, i == 1 ? term_unary : 0); // apply unary operator in last step
    if (i != factors.size() - 1)
      im1.release();
  }

  ASSERT(factors[0].type == _eval_factor_::LATTICE);
  return factors[0].vlattice;
}

void eval_general(std::vector<cgpt_Lattice_base*>& dst, std::vector<_eval_term_>& terms,int unary,bool ac) {

  // class A)
  // first separate all terms that are a pure linear combination:
  //   result_class_a = unary(A + B + C) taken using compatible_linear_combination

  // class B)
  // for all other terms, create terms and apply unary operators before summing
  //   result_class_b = unary(B*C*D) + unary(E*F)

  std::vector< std::vector<cgpt_lattice_term> > terms_a[NUM_FACTOR_UNARY], terms_b;

  for (size_t i=0;i<terms.size();i++) {
    auto& term = terms[i];
    ASSERT(term.factors.size() > 0);
    if (term.factors.size() == 1) {
      auto& factor = term.factors[0];
      ASSERT(factor.type == _eval_factor_::LATTICE);
      ASSERT(factor.unary >= 0 && factor.unary < NUM_FACTOR_UNARY);

      auto & a = terms_a[factor.unary];
      if (a.size() == 0) {
	a.resize(factor.vlattice.size());
      } else {
	ASSERT(a.size() == factor.vlattice.size());
      }

      for (int l=0;l<(int)factor.vlattice.size();l++) {
	a[l].push_back( cgpt_lattice_term( term.coefficient, factor.vlattice[l], false ) );
      }

    } else {

      auto factor_vlattice = eval_term(term.factors, unary);

      auto & a = terms_b;
      if (a.size() == 0) {
	a.resize(factor_vlattice.size());
      } else {
	ASSERT(a.size() == factor_vlattice.size());
      }

      for (int l=0;l<(int)factor_vlattice.size();l++) {
	a[l].push_back( cgpt_lattice_term( term.coefficient, factor_vlattice[l], true ) );
      }
    }
  }

  for (int j=0;j<NUM_FACTOR_UNARY;j++) {
    auto & a = terms_a[j];
    if (a.size() > 0) {
      if (dst.size() == 0)
	dst.resize(a.size(),0);
      ASSERT(dst.size() == a.size());

      bool mtrans = (j & BIT_TRANS) != 0;
      int singlet_rank = a[0][0].get_lat()->singlet_rank();
      int singlet_dim  = size_to_singlet_dim(singlet_rank, (int)a.size());

      if (singlet_rank == 2) {
	for (int r=0;r<singlet_dim;r++) {
	  for (int s=0;s<singlet_dim;s++) {
	    int idx1 = r*singlet_dim + s;
	    int idx2 = mtrans ? (s*singlet_dim + r) : idx1;
	    dst[idx1] = a[idx2][0].get_lat()->compatible_linear_combination(dst[idx1],ac, a[idx2], j, unary);
	  }
	}
      } else {
	for (int l=0;l<(int)a.size();l++) {
	  dst[l] = a[l][0].get_lat()->compatible_linear_combination(dst[l],ac, a[l], j, unary);
	}
      }
      ac=true;
    }
  }

  {
    auto & a = terms_b;
    if (a.size() > 0) {
      if (dst.size() == 0)
	dst.resize(a.size(),0);
      ASSERT(dst.size() == a.size());
      for (int l=0;l<(int)a.size();l++) {
	dst[l] = a[l][0].get_lat()->compatible_linear_combination(dst[l],ac, a[l], 0, 0); // unary operators have been applied above
      }
    }
  }

  for (auto& vterm : terms_b)
    for (auto& term : vterm)
      term.release();
}

static inline void simplify(_eval_term_& term) {
  auto& factors = term.factors;
  for (size_t i=0;i<factors.size()-1;) {
    if ((factors[i  ].type == _eval_factor_::GAMMA) &&
	(factors[i+1].type == _eval_factor_::GAMMA)) {
      factors[i].gamma = Gamma::mul[factors[i].gamma][factors[i+1].gamma];
      factors.erase(factors.begin()+i+1);
      continue;
    }
    i++;
  }
}

static void simplify(std::vector<_eval_term_>& terms) {
  for (size_t i=0;i<terms.size();i++) {
    auto& term = terms[i];
    simplify(term);
  }
}

EXPORT(eval,{

    PyObject*_dst, * _list,* _ac;
    int unary, idx;
    if (!PyArg_ParseTuple(args, "OOiOi", &_dst, &_list, &unary, &_ac, &idx)) {
      return NULL;
    }
    
    ASSERT(PyBool_Check(_ac));
    bool ac;
    cgpt_convert(_ac,ac);
    
    bool new_lattice = (Py_None == _dst);

    std::vector<_eval_term_> terms;
    eval_convert_factors(_list,terms,idx);
    
    std::vector<cgpt_Lattice_base*> dst;
    if (!new_lattice)
      cgpt_convert(_dst,dst);

    // do a first pass that, e.g., combines gamma algebra
    simplify(terms);

    // TODO: need a faster code path for dst = A*B*C*D*... ; in this case I could avoid one copy
    // if (expr_class_prod()) 
    //   eval_prod()
    // else
    
    // General code path:
    eval_general(dst,terms,unary,ac);

    if (new_lattice) {
      PyObject* ret = PyList_New(dst.size());
      for (long i=0;i<dst.size();i++)
	PyList_SetItem(ret,i,dst[i]->to_decl());
      return ret;
    }

    return PyLong_FromLong(0);
    
  });
