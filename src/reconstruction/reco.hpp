/**
 * @file reco.hpp
 * @brief Reconstruction chain shared by the CPU (Alpaka) and GPU (CUDA)
 *        programs. The backend-specific parts (algorithm types, queue/stream,
 *        memory resources, magnetic field) are provided by a Backend struct.
 */
#pragma once

// Project include(s).
#include "track_csv_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/examples/make_magnetic_field.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_gbts_seeding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/utils/propagation.hpp"

// System include(s).
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Output folder for the CSV files, set by CMake (see CMakeLists.txt).
#ifndef RECO_DATA_DIR
#define RECO_DATA_DIR "data"
#endif

namespace reco {

/**
 * @brief Runs the full reconstruction chain with the given backend.
 * @tparam Backend Struct providing the algorithm types, the queue/stream,
 *         the memory resources and the magnetic field for one platform.
 */
template <typename Backend>
int run(const traccc::opts::detector& detector_opts,
        const traccc::opts::magnetic_field& bfield_opts,
        const traccc::opts::input_data& input_opts,
        const traccc::opts::clusterization& clusterization_opts,
        const traccc::opts::track_seeding& seeding_opts,
        const traccc::opts::track_gbts_seeding& seeding_gbts_opts,
        const traccc::opts::track_finding& finding_opts,
        const traccc::opts::track_propagation& propagation_opts,
        const traccc::opts::track_fitting& fitting_opts,
        const traccc::opts::performance& performance_opts,
        const traccc::opts::accelerator& accelerator_opts,
        std::unique_ptr<const traccc::Logger> ilogger, bool usingGBTS) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Backend objects: queue/stream, memory resources, copy object.
    Backend be;
    auto& host_mr = be.host_mr();
    auto& device_mr = be.device_mr();
    auto& mr = be.mr();
    auto& copy = be.copy();
    auto& queue = be.queue();

    // Detector description (module designs and conditions), host and device.
    traccc::detector_design_description::host host_det_descr{host_mr};
    traccc::detector_conditions_description::host host_det_cond{host_mr};
    traccc::io::read_detector_description(
        host_det_descr, host_det_cond, detector_opts.detector_file,
        detector_opts.digitization_file, detector_opts.conditions_file,
        traccc::data_format::json);
    traccc::detector_design_description::buffer device_det_descr{
        [&]() {
            std::vector<unsigned int> sizes(host_det_descr.size());
            for (std::size_t i = 0; i < host_det_descr.size(); ++i) {
                auto this_design = host_det_descr.at(i);
                sizes[i] = std::max(
                    static_cast<unsigned int>(this_design.bin_edges_x().size()),
                    static_cast<unsigned int>(this_design.bin_edges_y().size()));
            }
            return sizes;
        }(),
        device_mr, &host_mr, vecmem::data::buffer_type::resizable};
    copy.setup(device_det_descr)->wait();
    copy(vecmem::get_data(host_det_descr), device_det_descr)->wait();

    traccc::detector_conditions_description::buffer device_det_cond{
        static_cast<traccc::detector_conditions_description::buffer::size_type>(
            host_det_cond.size()),
        device_mr};
    copy.setup(device_det_cond)->wait();
    copy(vecmem::get_data(host_det_cond), device_det_cond,
         vecmem::copy::type::host_to_device)
        ->wait();

    // Detray detector (geometry, material, surface grids), host and device.
    traccc::host_detector host_det;
    traccc::io::read_detector(host_det, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    const traccc::detector_buffer detector_buffer =
        traccc::buffer_from_host_detector(host_det, device_mr, copy);
    be.synchronize();

    // Output stats.
    std::uint64_t n_cells = 0;
    std::uint64_t n_measurements = 0;
    std::uint64_t n_spacepoints = 0;
    std::uint64_t n_seeds = 0;
    std::uint64_t n_found_tracks = 0;
    std::uint64_t n_fitted_tracks = 0;

    // Algorithm configuration(s).
    const traccc::gbts_seedfinder_config gbts_config(seeding_gbts_opts);
    const traccc::seedfinder_config seedfinder_config(seeding_opts);
    const traccc::seedfilter_config seedfilter_config(seeding_opts);
    const traccc::spacepoint_grid_config spacepoint_grid_config(seeding_opts);

    detray::propagation::config propagation_config(propagation_opts);

    traccc::finding_config finding_cfg(finding_opts);
    finding_cfg.propagation = propagation_config;

    traccc::fitting_config fitting_cfg(fitting_opts);
    fitting_cfg.propagation = propagation_config;

    traccc::track_params_estimation_config track_params_estimation_config;

    // Magnetic field: built on the host, then prepared by the backend.
    const auto host_field = traccc::details::make_magnetic_field(bfield_opts);
    const auto field = be.make_field(host_field, accelerator_opts);

    // Algorithms.
    typename Backend::clusterization ca(mr, copy, queue, clusterization_opts,
                                        logger().clone("ClusteringAlg"));
    typename Backend::meas_sorting ms(mr, copy, queue,
                                      logger().clone("MeasSortingAlg"));
    typename Backend::sp_formation sf(mr, copy, queue,
                                      logger().clone("SpFormationAlg"));
    typename Backend::triplet_seeding sa(seedfinder_config, spacepoint_grid_config,
                                         seedfilter_config, mr, copy, queue,
                                         logger().clone("SeedingAlg"));
    typename Backend::gbts_seeding gbts_sa(gbts_config, mr, copy, queue,
                                           logger().clone("GbtsSeedingAlg"));
    typename Backend::param_estim tp(track_params_estimation_config, mr, copy,
                                     queue, logger().clone("TrackParEstAlg"));
    typename Backend::finding finding_alg(finding_cfg, mr, copy, queue,
                                          logger().clone("FindingAlg"));
    typename Backend::fitting fitting_alg(fitting_cfg, mr, copy, queue,
                                          logger().clone("FittingAlg"));

    // Performance writer and timers.
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{},
        logger().clone("SeedingPerformanceWriter"));

    traccc::performance::timing_info elapsedTimes;

    // Per-event performance file: one row per event, with the size of the
    // event (cells, measurements, ...) and the time of each step.
    std::filesystem::create_directories(Backend::output_dir);
    const std::string perf_path = std::string(Backend::output_dir) + "/performance.csv";
    std::ofstream perf_file{perf_path};
    if (!perf_file) {
        throw std::runtime_error("Cannot open " + perf_path);
    }
    
    perf_file << "backend,event,n_cells,n_measurements,n_spacepoints,n_seeds,"
                 "n_found_tracks,n_fitted_tracks,t_read_ms,t_h2d_ms,"
                 "t_clusterization_ms,t_spacepoints_ms,t_seeding_ms,"
                 "t_params_ms,t_finding_ms,t_fitting_ms,t_wall_ms,t_d2h_ms\n";

    // Loop over events.
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::edm::measurement_collection::buffer measurements_buffer;
        traccc::edm::spacepoint_collection::buffer spacepoints_buffer;
        traccc::edm::seed_collection::buffer seeds_buffer;
        traccc::bound_track_parameters_collection_types::buffer params_buffer(
            0, *mr.host);
        traccc::edm::track_container<traccc::default_algebra>::buffer
            track_candidates_buffer;
        traccc::edm::track_container<traccc::default_algebra>::buffer
            track_states_buffer;

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            traccc::edm::silicon_cell_collection::host cells_per_event{host_mr};

            {
                traccc::performance::timer t("File reading", elapsedTimes);
                static constexpr bool DEDUPLICATE = true;
                traccc::io::read_cells(cells_per_event, event,
                                       input_opts.directory, logger().clone(),
                                       &host_det_cond, input_opts.format,
                                       DEDUPLICATE,
                                       input_opts.use_acts_geom_source);
            }

            n_cells += cells_per_event.size();

            // Copy the cells to the device.
            traccc::edm::silicon_cell_collection::buffer cells_buffer(
                static_cast<unsigned int>(cells_per_event.size()), mr.main);
            copy.setup(cells_buffer)->wait();
            copy(vecmem::get_data(cells_per_event), cells_buffer)->wait();

            {
                traccc::performance::timer t("Clusterization", elapsedTimes);
                auto unsorted_measurements =
                    ca(cells_buffer, device_det_descr, device_det_cond);
                measurements_buffer = ms(unsorted_measurements);
                be.synchronize();
            }

            {
                traccc::performance::timer t("Spacepoint formation",
                                             elapsedTimes);
                spacepoints_buffer = sf(detector_buffer, measurements_buffer);
                be.synchronize();
            }

            {
                traccc::performance::timer t("Seeding", elapsedTimes);
                if (usingGBTS) {
                    seeds_buffer = gbts_sa(spacepoints_buffer, measurements_buffer);
                } else {
                    seeds_buffer = sa(spacepoints_buffer);
                }
                be.synchronize();
            }

            {
                traccc::performance::timer t("Track params", elapsedTimes);
                params_buffer = tp(field, measurements_buffer, spacepoints_buffer,
                                   seeds_buffer);
                be.synchronize();
            }

            {
                traccc::performance::timer t("Track finding", elapsedTimes);
                track_candidates_buffer = finding_alg(
                    detector_buffer, field, measurements_buffer, params_buffer);
                be.synchronize();
            }

            {
                traccc::performance::timer t("Track fitting", elapsedTimes);
                track_states_buffer =
                    fitting_alg(detector_buffer, field, track_candidates_buffer);
                be.synchronize();
            }
        }  // Stop measuring wall time

        // Copy the results back to the host.
        traccc::edm::measurement_collection::host measurements_per_event{host_mr};
        traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
        traccc::edm::seed_collection::host seeds{host_mr};
        traccc::edm::track_collection<traccc::default_algebra>::host
            track_candidates{host_mr};
        traccc::edm::track_container<traccc::default_algebra>::host track_states{
            host_mr};

        copy(measurements_buffer, measurements_per_event)->wait();
        copy(spacepoints_buffer, spacepoints_per_event)->wait();
        copy(seeds_buffer, seeds)->wait();
        copy(track_candidates_buffer.tracks, track_candidates)->wait();
        copy(track_states_buffer.tracks, track_states.tracks)->wait();
        copy(track_states_buffer.states, track_states.states)->wait();
        be.synchronize();

        // Link the fitted tracks to the measurements of the event, then save them.
        track_states.measurements = vecmem::get_data(measurements_per_event);
        reco::write_tracks_csv(
            Backend::output_dir, event,
            traccc::edm::track_container<traccc::default_algebra>::const_data{
                track_states});

        // Statistics.
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_found_tracks += track_candidates.size();
        n_fitted_tracks += track_states.tracks.size();

        if (performance_opts.run) {
            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &host_det, input_opts.format, false);
            sd_performance_writer.write(vecmem::get_data(seeds),
                                        vecmem::get_data(spacepoints_per_event),
                                        vecmem::get_data(measurements_per_event),
                                        evt_data);
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    TRACCC_INFO("==> Statistics (" << Backend::name << ") ... ");
    TRACCC_INFO("- read    " << n_cells << " cells");
    TRACCC_INFO("- created " << n_measurements << " measurements");
    TRACCC_INFO("- created " << n_spacepoints << " spacepoints");
    TRACCC_INFO("- created " << n_seeds << " seeds");
    TRACCC_INFO("- found   " << n_found_tracks << " tracks");
    TRACCC_INFO("- fitted  " << n_fitted_tracks << " tracks");
    TRACCC_INFO("==> Elapsed times... " << elapsedTimes);

    return 0;
}

/**
 * @brief Parses the command line options and runs the chain with @p Backend.
 */
    template <typename Backend>
    int main(int argc, char* argv[]) {
        std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
            std::string("TracccReco_") + Backend::name, traccc::Logging::Level::INFO);

        traccc::opts::detector detector_opts;
        traccc::opts::magnetic_field bfield_opts;
        traccc::opts::input_data input_opts;
        traccc::opts::clusterization clusterization_opts;
        traccc::opts::track_seeding seeding_opts;
        traccc::opts::track_gbts_seeding seeding_gbts_opts;
        traccc::opts::track_finding finding_opts;
        traccc::opts::track_propagation propagation_opts;
        traccc::opts::track_fitting fitting_opts;
        traccc::opts::performance performance_opts;
        traccc::opts::accelerator accelerator_opts;
        traccc::opts::program_options program_opts{
            std::string("Full Tracking Chain (") + Backend::name + ")",
            {detector_opts, bfield_opts, input_opts, clusterization_opts,
            seeding_opts, seeding_gbts_opts, finding_opts, propagation_opts,
            performance_opts, fitting_opts, accelerator_opts},
            argc,
            argv,
            logger->cloneWithSuffix("Options")};

        return run<Backend>(detector_opts, bfield_opts, input_opts,
                            clusterization_opts, seeding_opts, seeding_gbts_opts,
                            finding_opts, propagation_opts, fitting_opts,
                            performance_opts, accelerator_opts, logger->clone(),
                            seeding_gbts_opts.useGBTS);
    }

} 
