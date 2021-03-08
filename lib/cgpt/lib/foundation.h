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

    
    This file tries to isolate foundational code of the data parallel layer.
    (Some of this could move to Grid.)
*/
#include <Grid/Grid.h>

using namespace Grid;

#if defined(GRID_SYCL) || defined(GRID_CUDA) || defined(GRID_HIP)
#define GRID_HAS_ACCELERATOR
#endif

#if defined (GRID_COMMS_MPI3)
#define CGPT_USE_MPI 1
#endif

#define VECTOR_VIEW_OPEN(l,v,mode)				\
  Vector< decltype(l[0].View(mode)) > v; v.reserve(l.size());	\
  for(uint64_t k=0;k<l.size();k++)				\
    v.push_back(l[k].View(mode));

#define VECTOR_VIEW_OPEN_POINTER(l,v,p,mode)			\
  Vector< decltype(l[0].View(mode)) > v; v.reserve(l.size());	\
  for(uint64_t k=0;k<l.size();k++)				\
    v.push_back(l[k].View(mode));	                        \
  typename std::remove_reference<decltype(v[0])>::type* p = &v[0];

#define VECTOR_VIEW_CLOSE(v)				\
  for(uint64_t k=0;k<v.size();k++) v[k].ViewClose();

#define VECTOR_VIEW_CLOSE_POINTER(v,p)                  \
  for(uint64_t k=0;k<v.size();k++) v[k].ViewClose();    \
  p = nullptr;

// use grid's print prefix but more conveniently
#define grid_message(...) \
  { \
    char _buf[1024]; \
    sprintf(_buf, __VA_ARGS__); \
    std::cout << GridLogMessage << _buf; \
    fflush(stdout);                      \
  }

NAMESPACE_BEGIN(Grid);

// aligned vector
template<class T> using AlignedVector = std::vector<T,alignedAllocator<T> >;

#if defined(GRID_CUDA)||defined(GRID_HIP)
#include "foundation/reduce_gpu.h"
#endif

#include "foundation/reduce.h"
#include "foundation/unary.h"
#include "foundation/binary.h"
#include "foundation/ternary.h"
#include "foundation/et.h"
#include "foundation/block.h"
#include "foundation/transfer.h"
#include "foundation/basis.h"
#include "foundation/eigen.h"
#include "foundation/matrix.h"
#include "foundation/clover.h"
#include "foundation/coarse.h"


NAMESPACE_END(Grid);
