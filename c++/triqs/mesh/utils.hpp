// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#pragma once

#include <itertools/itertools.hpp>
#include <nda/nda.hpp>
#include <triqs/utility/tuple_tools.hpp>

#include <fmt/core.h>

#include "./concepts.hpp"

namespace triqs::mesh {

  using dcomplex = std::complex<double>;

  using nda::array;
  using nda::array_view;
  using nda::ellipsis;
  using nda::eye;
  using nda::matrix;
  using nda::matrix_const_view;
  using nda::matrix_view;
  using nda::range;

  //-------------------------------------------------------------------------

  template <typename... T> uint64_t hash(T &&...ts) { return (std::hash<std::decay_t<T>>()(ts) + ...); }

  //------------------------------------------------------------------------

  inline long positive_modulo(long r, long d) {
    long res = r % d;
    return (res >= 0 ? res : res + d);
  }

  //------------------------------------------------------------------------

  /// Given a mesh, return an array of the values of the mesh
  template <Mesh M> [[nodiscard]] nda::vector<typename M::value_t> values(M const &m) {
    auto res = nda::vector<typename M::value_t>(m.size());
    for (auto i : range(m.size())) res(i) = m[i].value();
    return res;
  }

  //------------------------------------------------------------------------

  template <Mesh M> static constexpr bool is_product  = false;
  template <Mesh Ms> static constexpr int n_variables = 1;

  /// A place holder for : or *all*
  using all_t = nda::range::all_t;

  /** The statistics : Boson or Fermion */
  enum statistic_enum { Boson = 0, Fermion = 1 };

  // 1 for Boson, -1 for Fermion
  inline int sign(statistic_enum s) { return (s == Boson ? 1 : -1); }

  /// Boson*Fermion -> Fermion, others -> Boson
  inline statistic_enum operator*(statistic_enum i, statistic_enum j) { return (i == j ? Boson : Fermion); }

  //------------------------------ closest_mesh_pt -------------------------

  template <typename T> struct closest_mesh_point_t {
    T value;
  };

  // Returns a lazy structure containing x that will be used by the [] operator of gf e.g.
  template <typename... T> auto closest_mesh_pt(T &&...x) {
    if constexpr (sizeof...(T) == 1)
      return closest_mesh_point_t<std::decay_t<T>...>{std::forward<T>(x)...};
    else
      return std::tuple{closest_mesh_point_t<std::decay_t<T>>{std::forward<T>(x)}...};
  }

  namespace details {

    // FIXME use ranges::views, but clang > 15 only
    auto sum_to_regular_chunk(auto const &R, auto f) {
      auto it  = std::begin(R);
      auto e   = std::end(R);
      auto res = make_regular(f(*it));
      for (++it; it != e; ++it) res += f(*it);
      return res;
    }

    // FIXME use ranges::views, but clang > 15 only
    auto sum_to_regular(auto const &R, auto f) {
      auto it = std::begin(R), e = std::end(R);
      // return sum_to_regular_chunk(R, f);
      const size_t n = std::distance(it, e);

      constexpr auto vec_size = 8;
      if (n < vec_size) return sum_to_regular_chunk(R, std::move(f));
      const auto evaluate = [f](const auto x) { return make_regular(f(*x)); };
      using result_type   = std::invoke_result_t<decltype(evaluate), decltype(it)>;
      alignas(64) std::array<result_type, vec_size> results{};

#pragma clang loop unroll(full) vectorize(enable)
#pragma GCC ivdep unroll(vec_size)
      for (auto i = 0; i < vec_size; ++i) { results[i] = evaluate(std::next(it, i)); }
      std::advance(it, vec_size); // Move the iterator forward by vec_size

      for (auto i = vec_size; i < (n & -vec_size); i += vec_size) {
#pragma GCC ivdep unroll(vec_size)
#pragma clang loop unroll(full) vectorize(enable)
        for (auto j = 0; j < vec_size; ++j) { results[j] += f(*std::next(it, j)); }
        std::advance(it, vec_size); // Move the iterator forward by vec_size
      }

      result_type res = results[0];
      for (auto i = 1; i < vec_size; ++i) { res += results[i]; }

      for (auto i = n & (-vec_size); i < n; ++i) { res += f(*it++); }
      return res;
    }

  } // namespace details

} // namespace triqs::mesh
