// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/**
 * @file reco_cpu.cpp
 * @brief Reconstruction chain run on the CPU through the Alpaka algorithms.
 *        The chain itself is in reco.hpp; this file only defines the backend.
 */

// Project include(s).
#include "reco.hpp"
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/alpaka/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/alpaka/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/seed_parameter_estimation_algorithm.hpp"
#include "traccc/alpaka/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/triplet_seeding_algorithm.hpp"
#include "traccc/alpaka/utils/queue.hpp"
#include "traccc/alpaka/utils/vecmem_objects.hpp"

/// Backend running the traccc Alpaka algorithms.
struct alpaka_backend {
    static constexpr const char* name = "alpaka";
    static constexpr const char* output_dir = RECO_DATA_DIR "/reco_cpu";

    // Algorithm types.
    using clusterization = traccc::alpaka::clusterization_algorithm;
    using meas_sorting = traccc::alpaka::measurement_sorting_algorithm;
    using sp_formation =
        traccc::alpaka::silicon_pixel_spacepoint_formation_algorithm;
    using triplet_seeding = traccc::alpaka::triplet_seeding_algorithm;
    using gbts_seeding = traccc::alpaka::gbts_seeding_algorithm;
    using param_estim = traccc::alpaka::seed_parameter_estimation_algorithm;
    using finding = traccc::alpaka::combinatorial_kalman_filter_algorithm;
    using fitting = traccc::alpaka::kalman_fitting_algorithm;

    // Members are constructed top to bottom: the queue must come first.
    traccc::alpaka::queue m_queue;
    traccc::alpaka::vecmem_objects m_vo{m_queue};
    traccc::memory_resource m_mr{m_vo.device_mr(), &m_vo.host_mr()};

    traccc::alpaka::queue& queue() { return m_queue; }
    vecmem::copy& copy() { return m_vo.async_copy(); }
    vecmem::memory_resource& host_mr() { return m_vo.host_mr(); }
    vecmem::memory_resource& device_mr() { return m_vo.device_mr(); }
    traccc::memory_resource& mr() { return m_mr; }
    void synchronize() { m_queue.synchronize(); }

    /// The Alpaka algorithms use the host magnetic field as it is.
    template <typename F>
    F make_field(const F& host_field, const traccc::opts::accelerator&) {
        return host_field;
    }
};

int main(int argc, char* argv[]) {
    return reco::main<alpaka_backend>(argc, argv);
}