// Copyright (c) 2022-2023 Simons Foundation
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
// Authors: Nils Wentzell

#include <benchmark/benchmark.h>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

// ===== Interpolate DLR ImTime Green function

constexpr nda::clef::placeholder<2> tau_;
using namespace triqs::gfs;

auto onefermion(auto tau, double eps, double beta) { return -exp(-eps * tau) / (1 + exp(-beta * eps)); }

static void GfDLREval(benchmark::State &state) {

  double beta  = 2.0;
  double w_max = 5.0;
  double eps   = 1e-10;
  double omega = 1.337;

  auto mesh = dlr_imtime{beta, Fermion, w_max, eps};
  auto G    = gf<dlr_imtime, scalar_valued>{mesh};
  G[tau_] << onefermion(tau_, omega, beta);
  auto G_dlr_coeff = make_gf_dlr(G);

  for (auto _ : state) {
    for (auto tau : mesh) benchmark::DoNotOptimize(G_dlr_coeff(double(tau)));
  }
}
BENCHMARK(GfDLREval)->RangeMultiplier(2)->Range(1024, 8192);

BENCHMARK_MAIN();
