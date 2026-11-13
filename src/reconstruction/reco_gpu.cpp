// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/**
 * @file reco_gpu.cpp
 * @brief Reconstruction chain run on the GPU through the CUDA algorithms.
 *        The chain itself is in reco.hpp; this file only defines the backend.
 */

// Project include(s).
#include "reco.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/cuda/seeding/seed_parameter_estimation_algorithm.hpp"
#include "traccc/cuda/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/triplet_seeding_algorithm.hpp"
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/cuda/utils/stream_wrapper.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/stream_wrapper.hpp>

/// Backend running the traccc CUDA algorithms. Every thread of reco.hpp
/// creates its own instance, i.e. its own CUDA stream.
struct cuda_backend {
    static constexpr const char* name = "cuda";
    static constexpr const char* output_dir = RECO_DATA_DIR "/reco_gpu";

    // Algorithm types.
    using clusterization = traccc::cuda::clusterization_algorithm;
    using meas_sorting = traccc::cuda::measurement_sorting_algorithm;
    using sp_formation =
        traccc::cuda::silicon_pixel_spacepoint_formation_algorithm;
    using triplet_seeding = traccc::cuda::triplet_seeding_algorithm;
    using gbts_seeding = traccc::cuda::gbts_seeding_algorithm;
    using param_estim = traccc::cuda::seed_parameter_estimation_algorithm;
    using finding = traccc::cuda::combinatorial_kalman_filter_algorithm;
    using fitting = traccc::cuda::kalman_fitting_algorithm;

    // Members are constructed top to bottom: stream, copy, then memory.
    vecmem::cuda::stream_wrapper m_vecmem_stream;
    traccc::cuda::stream_wrapper m_stream{m_vecmem_stream.stream()};
    vecmem::cuda::async_copy m_copy{m_stream.cudaStream()};
    vecmem::host_memory_resource m_host_mr;
    vecmem::cuda::host_memory_resource m_pinned_mr;  // pinned host memory
    vecmem::cuda::device_memory_resource m_device_mr;
    traccc::memory_resource m_mr{m_device_mr, &m_pinned_mr};

    traccc::cuda::stream_wrapper& queue() { return m_stream; }
    vecmem::copy& copy() { return m_copy; }
    vecmem::memory_resource& host_mr() { return m_host_mr; }
    vecmem::memory_resource& device_mr() { return m_device_mr; }
    traccc::memory_resource& mr() { return m_mr; }
    void synchronize() { m_stream.synchronize(); }

    /// Copies the magnetic field to the GPU (texture or global memory).
    template <typename F>
    auto make_field(const F& host_field,
                    const traccc::opts::accelerator& accelerator_opts) {
        return traccc::cuda::make_magnetic_field(
            host_field,
            accelerator_opts.use_gpu_texture_memory
                ? traccc::cuda::magnetic_field_storage::texture_memory
                : traccc::cuda::magnetic_field_storage::global_memory);
    }
};

int main(int argc, char* argv[]) {
    return reco::main<cuda_backend>(argc, argv);
}