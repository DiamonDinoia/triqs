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
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Implementation details for triqs/mc_tools/random_generator.hpp.
 */

#include "./random_generator.hpp"
#include "../utility/first_include.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <simdrng/chacha_simd.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <thread>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace triqs::mc_tools {

  // Names of the supported engines; the empty string is additionally accepted as an alias for mt19937_64.
  static const std::vector<std::string> engine_names = {"mt19937_64", "mt19937", "ranlux48", "ranlux24", "minstd_rand", "knuth_b", "chacha"};

  namespace {
    // Adaptor over simdrng's ChaCha8 that adds the iostream `<<`/`>>` operators the standard-library
    // engines already provide, so the generic rng_model wraps and serializes it uniformly. Serialization
    // mirrors simdrng's own state contract: the ChaCha matrix (16 u32), the result cache (8 u64) and the
    // cache index, as whitespace-separated integers. After a refill the cache index sits at 8 (exhausted),
    // so a restored engine resumes generating at the correct block. Seeding, drawing and refill are
    // inherited unchanged; the SIMD batching lives inside ChaCha8Native, so the rng_model scalar refill
    // loop still vectorizes.
    struct chacha_engine : simdrng::ChaCha8Native {
      using simdrng::ChaCha8Native::ChaCha8Native;

      friend std::ostream &operator<<(std::ostream &os, chacha_engine const &e) {
        for (auto w : e.getStateForSerde()) os << w << ' ';
        for (auto v : e.result_cache()) os << v << ' ';
        return os << static_cast<unsigned>(e.result_index());
      }
      friend std::istream &operator>>(std::istream &is, chacha_engine &e) {
        matrix_type matrix{};
        for (auto &w : matrix) is >> w;
        result_cache_type cache{};
        for (auto &v : cache) is >> v;
        unsigned idx = 0;
        is >> idx;
        e.setState(matrix);
        e.set_result_cache(cache);
        e.set_result_index(static_cast<std::uint8_t>(idx));
        return is;
      }
    };
  } // namespace

  random_generator::random_generator(std::string name, std::uint64_t seed, mpi::communicator c)
     : buffer_(buffer_size), name_(std::move(name)) {
    // random_generator is not thread-safe and seeds streams by MPI rank, not by thread: it exposes no
    // per-thread stream, so every instance must be constructed on a single thread. We capture the
    // thread of the first construction (the function-local static is initialized exactly once, in a
    // thread-safe manner) and require every later construction to be on that same thread. This catches
    // instantiating RNGs on multiple threads (e.g. one per worker thread, or inside a parallel region),
    // which would otherwise silently produce identical (same seed+rank) or racy streams. Cheap and
    // always on: construction is a cold path. (Drawing off the owning thread is caught separately by
    // check_thread.) This is portable across OpenMP, std::thread and pthread -- no parallel-region query.
    static std::thread::id const ctor_thread = std::this_thread::get_id();
    if (std::this_thread::get_id() != ctor_thread)
      throw std::runtime_error("Error in random_generator: all instances must be constructed on the same thread; "
                               "random_generator is not thread-safe and provides no per-thread stream");
    if (c.size() > 1) { // collective: all ranks of c construct together and must agree on the seed
      auto lo = mpi::all_reduce(seed, c, MPI_MIN);
      auto hi = mpi::all_reduce(seed, c, MPI_MAX);
      if (lo != hi) throw std::runtime_error("Error in random_generator: all MPI ranks must pass the same seed");
    }
    std::array<std::uint64_t, 1> key{static_cast<std::uint64_t>(c.rank())};
    initialize_rng(name_, seed, key); // {rank} span: decorrelated stream per rank
    refill();
  }

  void random_generator::initialize_rng(std::string const &name, std::uint64_t seed, std::span<std::uint64_t const> spawn_key) {

    // chacha: simdrng counter-based generator. Its own splitmix64 expands the seed into the key; the
    // spawn key (the MPI rank) rides in as the ChaCha nonce, giving decorrelated per-rank streams
    // without splitmix_seed_seq.
    if (name == "chacha") {
      std::uint64_t const nonce = spawn_key.empty() ? 0 : spawn_key.front();
      ptr_                      = std::make_unique<rng_model<chacha_engine>>(chacha_engine{seed, 0, nonce});
      return;
    }

    // All standard-library engines have their full state initialized from the (seed, spawn_key) pair.
    auto sseq = splitmix_seed_seq{seed, spawn_key};

    // mt19937_64: native 64-bit engine (default)
    if (name.empty() || name == "mt19937_64") {
      ptr_ = std::make_unique<rng_model<std::mt19937_64>>(std::mt19937_64{sseq});
      return;
    }

    // Wrap an engine with native output < 64 bits using independent_bits_engine.
    auto wrap = [this](auto base) {
      using engine_t = std::independent_bits_engine<decltype(base), 64, std::uint64_t>;
      ptr_           = std::make_unique<rng_model<engine_t>>(engine_t{std::move(base)});
    };

    if (name == "mt19937") return wrap(std::mt19937{sseq});         // 32-bit Mersenne Twister
    if (name == "ranlux48") return wrap(std::ranlux48{sseq});       // 48-bit
    if (name == "ranlux24") return wrap(std::ranlux24{sseq});       // 24-bit
    if (name == "minstd_rand") return wrap(std::minstd_rand{sseq}); // 31-bit LCG
    if (name == "knuth_b") return wrap(std::knuth_b{sseq});         // 31-bit shuffle engine

    throw std::runtime_error(fmt::format("Error in random_generator::initialize_rng: RNG with name '{}' is not supported", name));
  }

  std::string random_generator_names(std::string const &sep) { return fmt::format("{}", fmt::join(engine_names, sep)); }

  std::vector<std::string> random_generator_names_list() { return engine_names; }

} // namespace triqs::mc_tools
