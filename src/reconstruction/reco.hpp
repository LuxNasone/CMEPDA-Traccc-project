/**
 * @file reco.hpp
 * @brief Reconstruction chain shared by the CPU (Alpaka) and GPU (CUDA)
 *        programs. The backend-specific parts (algorithm types, queue/stream,
 *        memory resources, magnetic field) are provided by a Backend struct.
 *
 * Events can be processed in parallel with --threads=N: every thread owns a
 * complete copy of the backend (its own queue/stream, memory, device detector
 * and algorithms) and takes the next free event from a shared counter. With
 * N=1 the behaviour is the same as the original sequential example.
 */
#pragma once

// Project include(s).
#include "track_csv_writer.hpp"
// reco.hpp is included before the backend headers, so every traccc type used
// by the template must be declared here (non-dependent names are looked up
// where the template is defined, not where it is instantiated).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/examples/make_magnetic_field.hpp"
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
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// Output folder for the CSV files, set by CMake (see CMakeLists.txt).
#ifndef RECO_DATA_DIR
#define RECO_DATA_DIR "data"
#endif

namespace reco {

/// Times of the reconstruction steps of one event, in milliseconds.
struct event_times {
    double read = 0.;            ///< Reading the cells from file
    double h2d = 0.;             ///< Copying the cells to the device
    double clusterization = 0.;  ///< Clusterization + measurement sorting
    double spacepoints = 0.;     ///< Spacepoint formation
    double seeding = 0.;         ///< Seeding (triplet or GBTS)
    double params = 0.;          ///< Track parameter estimation
    double finding = 0.;         ///< Track finding (CKF)
    double fitting = 0.;         ///< Track fitting (Kalman)
    double wall = 0.;            ///< From reading to fitting (all of the above)
    double d2h = 0.;             ///< Copying the results back to the host
};

/// Size, times and scheduling information of one processed event.
struct event_result {
    std::size_t event = 0;           ///< Event number
    unsigned int thread = 0;         ///< Thread (and stream) that processed it
    std::size_t n_cells = 0;
    std::size_t n_measurements = 0;
    std::size_t n_spacepoints = 0;
    std::size_t n_seeds = 0;
    std::size_t n_found_tracks = 0;
    std::size_t n_fitted_tracks = 0;
    event_times times;
    double t_start = 0.;  ///< Start, in ms since the start of the event loop
    double t_end = 0.;    ///< End (after writing), same reference
};

using clock_type = std::chrono::steady_clock;

/// Milliseconds elapsed from @p from to @p to.
inline double ms_between(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

/// Measures the time spent in a scope and stores it, in ms, in @p out.
class scope_timer {
   public:
    explicit scope_timer(double& out) : m_out(out), m_start(clock_type::now()) {}
    ~scope_timer() { m_out = ms_between(m_start, clock_type::now()); }
    scope_timer(const scope_timer&) = delete;
    scope_timer& operator=(const scope_timer&) = delete;

   private:
    double& m_out;
    clock_type::time_point m_start;
};

/// Type of the host magnetic field.
using host_field_t = decltype(traccc::details::make_magnetic_field(
    std::declval<const traccc::opts::magnetic_field&>()));

/**
 * @brief Everything read or built once and then only read by all the threads:
 *        options, detector description, geometry, configurations, field.
 */
struct shared_inputs {
    const traccc::opts::input_data& input_opts;
    const traccc::opts::clusterization& clusterization_opts;
    const traccc::opts::accelerator& accelerator_opts;
    bool use_gbts;

    traccc::detector_design_description::host& det_descr;
    traccc::detector_conditions_description::host& det_cond;
    traccc::host_detector& detector;
    const host_field_t& field;

    const traccc::gbts_seedfinder_config& gbts_config;
    const traccc::seedfinder_config& seedfinder_config;
    const traccc::seedfilter_config& seedfilter_config;
    const traccc::spacepoint_grid_config& grid_config;
    const traccc::track_params_estimation_config& params_config;
    const traccc::finding_config& finding_config;
    const traccc::fitting_config& fitting_config;

    /// Seeding performance writer, or nullptr when it is not requested.
    /// It is not thread safe: it is only used while holding perf_mutex.
    traccc::seeding_performance_writer* perf_writer;
    std::mutex& perf_mutex;
};

/// Number of bins of every module design, used to size the device buffer.
inline std::vector<unsigned int> design_sizes(
    traccc::detector_design_description::host& det_descr) {
    std::vector<unsigned int> sizes(det_descr.size());
    for (std::size_t i = 0; i < det_descr.size(); ++i) {
        auto this_design = det_descr.at(i);
        sizes[i] = std::max(
            static_cast<unsigned int>(this_design.bin_edges_x().size()),
            static_cast<unsigned int>(this_design.bin_edges_y().size()));
    }
    return sizes;
}

/**
 * @brief One complete reconstruction chain on its own queue/stream: backend
 *        objects, device copies of the detector, field and algorithms.
 *        Every thread owns one worker, so workers never share mutable state.
 */
template <typename Backend>
class worker {
   public:
    using field_t = decltype(std::declval<Backend&>().make_field(
        std::declval<const host_field_t&>(),
        std::declval<const traccc::opts::accelerator&>()));

    worker(shared_inputs& in, const traccc::Logger& log)
        : m_in(in),
          m_det_descr(design_sizes(in.det_descr), m_be.device_mr(),
                      &m_be.host_mr(), vecmem::data::buffer_type::resizable),
          m_det_cond(static_cast<traccc::detector_conditions_description::
                                     buffer::size_type>(in.det_cond.size()),
                     m_be.device_mr()),
          m_detector(traccc::buffer_from_host_detector(
              in.detector, m_be.device_mr(), m_be.copy())),
          m_field(m_be.make_field(in.field, in.accelerator_opts)),
          m_ca(m_be.mr(), m_be.copy(), m_be.queue(), in.clusterization_opts,
               log.clone("ClusteringAlg")),
          m_ms(m_be.mr(), m_be.copy(), m_be.queue(),
               log.clone("MeasSortingAlg")),
          m_sf(m_be.mr(), m_be.copy(), m_be.queue(),
               log.clone("SpFormationAlg")),
          m_sa(in.seedfinder_config, in.grid_config, in.seedfilter_config,
               m_be.mr(), m_be.copy(), m_be.queue(), log.clone("SeedingAlg")),
          m_gbts_sa(in.gbts_config, m_be.mr(), m_be.copy(), m_be.queue(),
                    log.clone("GbtsSeedingAlg")),
          m_tp(in.params_config, m_be.mr(), m_be.copy(), m_be.queue(),
               log.clone("TrackParEstAlg")),
          m_finding(in.finding_config, m_be.mr(), m_be.copy(), m_be.queue(),
                    log.clone("FindingAlg")),
          m_fitting(in.fitting_config, m_be.mr(), m_be.copy(), m_be.queue(),
                    log.clone("FittingAlg")),
          m_log(log.clone()) {
        auto& copy = m_be.copy();
        copy.setup(m_det_descr)->wait();
        copy(vecmem::get_data(in.det_descr), m_det_descr)->wait();
        copy.setup(m_det_cond)->wait();
        copy(vecmem::get_data(in.det_cond), m_det_cond,
             vecmem::copy::type::host_to_device)
            ->wait();
        m_be.synchronize();
    }

    /// Reconstructs one event, writes its tracks and returns its numbers.
    event_result process(std::size_t event, clock_type::time_point loop_start) {
        auto& host_mr = m_be.host_mr();
        auto& mr = m_be.mr();
        auto& copy = m_be.copy();

        event_result res;
        res.event = event;
        res.t_start = ms_between(loop_start, clock_type::now());
        event_times& times = res.times;

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
            scope_timer wall_st(times.wall);

            traccc::edm::silicon_cell_collection::host cells_per_event{host_mr};
            {
                scope_timer st(times.read);
                static constexpr bool DEDUPLICATE = true;
                traccc::io::read_cells(
                    cells_per_event, event, m_in.input_opts.directory,
                    m_log->clone(), &m_in.det_cond, m_in.input_opts.format,
                    DEDUPLICATE, m_in.input_opts.use_acts_geom_source);
            }
            res.n_cells = cells_per_event.size();

            // Copy the cells to the device.
            traccc::edm::silicon_cell_collection::buffer cells_buffer(
                static_cast<unsigned int>(cells_per_event.size()), mr.main);
            {
                scope_timer st(times.h2d);
                copy.setup(cells_buffer)->wait();
                copy(vecmem::get_data(cells_per_event), cells_buffer)->wait();
            }

            // Every step synchronizes only the stream of this worker: the
            // other threads keep running.
            {
                scope_timer st(times.clusterization);
                auto unsorted_measurements =
                    m_ca(cells_buffer, m_det_descr, m_det_cond);
                measurements_buffer = m_ms(unsorted_measurements);
                m_be.synchronize();
            }
            {
                scope_timer st(times.spacepoints);
                spacepoints_buffer = m_sf(m_detector, measurements_buffer);
                m_be.synchronize();
            }
            {
                scope_timer st(times.seeding);
                if (m_in.use_gbts) {
                    seeds_buffer =
                        m_gbts_sa(spacepoints_buffer, measurements_buffer);
                } else {
                    seeds_buffer = m_sa(spacepoints_buffer);
                }
                m_be.synchronize();
            }
            {
                scope_timer st(times.params);
                params_buffer = m_tp(m_field, measurements_buffer,
                                     spacepoints_buffer, seeds_buffer);
                m_be.synchronize();
            }
            {
                scope_timer st(times.finding);
                track_candidates_buffer = m_finding(
                    m_detector, m_field, measurements_buffer, params_buffer);
                m_be.synchronize();
            }
            {
                scope_timer st(times.fitting);
                track_states_buffer =
                    m_fitting(m_detector, m_field, track_candidates_buffer);
                m_be.synchronize();
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
        {
            scope_timer st(times.d2h);
            copy(measurements_buffer, measurements_per_event)->wait();
            copy(spacepoints_buffer, spacepoints_per_event)->wait();
            copy(seeds_buffer, seeds)->wait();
            copy(track_candidates_buffer.tracks, track_candidates)->wait();
            copy(track_states_buffer.tracks, track_states.tracks)->wait();
            copy(track_states_buffer.states, track_states.states)->wait();
            m_be.synchronize();
        }

        // Link the fitted tracks to the measurements of the event, then save
        // them. Every event has its own files, so threads never collide.
        track_states.measurements = vecmem::get_data(measurements_per_event);
        reco::write_tracks_csv(
            Backend::output_dir, event,
            traccc::edm::track_container<traccc::default_algebra>::const_data{
                track_states});

        res.n_measurements = measurements_per_event.size();
        res.n_spacepoints = spacepoints_per_event.size();
        res.n_seeds = seeds.size();
        res.n_found_tracks = track_candidates.size();
        res.n_fitted_tracks = track_states.tracks.size();

        if (m_in.perf_writer != nullptr) {
            traccc::event_data evt_data(
                m_in.input_opts.directory, event, host_mr,
                m_in.input_opts.use_acts_geom_source, &m_in.detector,
                m_in.input_opts.format, false);
            const std::lock_guard<std::mutex> lock(m_in.perf_mutex);
            m_in.perf_writer->write(vecmem::get_data(seeds),
                                    vecmem::get_data(spacepoints_per_event),
                                    vecmem::get_data(measurements_per_event),
                                    evt_data);
        }

        res.t_end = ms_between(loop_start, clock_type::now());
        return res;
    }

   private:
    // Members are constructed top to bottom: the backend must come first.
    shared_inputs& m_in;
    Backend m_be;
    traccc::detector_design_description::buffer m_det_descr;
    traccc::detector_conditions_description::buffer m_det_cond;
    traccc::detector_buffer m_detector;
    field_t m_field;

    typename Backend::clusterization m_ca;
    typename Backend::meas_sorting m_ms;
    typename Backend::sp_formation m_sf;
    typename Backend::triplet_seeding m_sa;
    typename Backend::gbts_seeding m_gbts_sa;
    typename Backend::param_estim m_tp;
    typename Backend::finding m_finding;
    typename Backend::fitting m_fitting;

    std::unique_ptr<const traccc::Logger> m_log;
};

/**
 * @brief Runs the full reconstruction chain with the given backend.
 * @tparam Backend Struct providing the algorithm types, the queue/stream,
 *         the memory resources and the magnetic field for one platform.
 * @param n_threads Number of events processed at the same time.
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
        std::unique_ptr<const traccc::Logger> ilogger, bool usingGBTS,
        unsigned int n_threads) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Host memory for everything read once and shared by the threads.
    vecmem::host_memory_resource host_mr;

    // Detector description (module designs and conditions).
    traccc::detector_design_description::host host_det_descr{host_mr};
    traccc::detector_conditions_description::host host_det_cond{host_mr};
    traccc::io::read_detector_description(
        host_det_descr, host_det_cond, detector_opts.detector_file,
        detector_opts.digitization_file, detector_opts.conditions_file,
        traccc::data_format::json);

    // Detray detector (geometry, material, surface grids).
    traccc::host_detector host_det;
    traccc::io::read_detector(host_det, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Algorithm configuration(s): the same for every thread.
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

    // Magnetic field on the host; every worker prepares its own copy.
    const host_field_t host_field =
        traccc::details::make_magnetic_field(bfield_opts);

    // Seeding performance writer (optional, shared under a mutex).
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{},
        logger().clone("SeedingPerformanceWriter"));
    std::mutex perf_mutex;

    shared_inputs in{input_opts,
                     clusterization_opts,
                     accelerator_opts,
                     usingGBTS,
                     host_det_descr,
                     host_det_cond,
                     host_det,
                     host_field,
                     gbts_config,
                     seedfinder_config,
                     seedfilter_config,
                     spacepoint_grid_config,
                     track_params_estimation_config,
                     finding_cfg,
                     fitting_cfg,
                     performance_opts.run ? &sd_performance_writer : nullptr,
                     perf_mutex};

    // The output folder is created before the threads start, so that they
    // never try to create it at the same time.
    std::filesystem::create_directories(Backend::output_dir);

    // Events to process, and one result slot per event: every thread writes
    // only the slots of its own events, so no lock is needed.
    std::vector<std::size_t> events;
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {
        events.push_back(event);
    }
    std::vector<event_result> results(events.size());

    n_threads = std::max(1u, std::min<unsigned int>(
                                 n_threads, static_cast<unsigned int>(
                                                std::max<std::size_t>(
                                                    events.size(), 1u))));
    TRACCC_INFO("Processing " << events.size() << " events with " << n_threads
                              << " thread(s)");

    std::atomic<std::size_t> next_event{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    // The clock of the event loop starts when every worker is ready, so the
    // set-up of the workers (device copies of the detector) is not counted.
    std::latch ready(n_threads);
    std::latch go(1);
    clock_type::time_point loop_start;

    const auto record_error = [&]() {
        const std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
            first_error = std::current_exception();
        }
        failed = true;
    };

    const auto work = [&](unsigned int thread_id) {
        std::unique_ptr<worker<Backend>> w;
        try {
            w = std::make_unique<worker<Backend>>(in, logger());
        } catch (...) {
            record_error();
        }
        ready.count_down();
        go.wait();
        if (!w) {
            return;
        }
        try {
            while (!failed) {
                const std::size_t i = next_event.fetch_add(1);
                if (i >= events.size()) {
                    break;
                }
                results[i] = w->process(events[i], loop_start);
                results[i].thread = thread_id;
            }
        } catch (...) {
            record_error();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int t = 0; t < n_threads; ++t) {
        threads.emplace_back(work, t);
    }
    ready.wait();
    loop_start = clock_type::now();
    go.count_down();
    for (auto& t : threads) {
        t.join();
    }
    const double loop_ms = ms_between(loop_start, clock_type::now());

    if (first_error) {
        std::rethrow_exception(first_error);
    }
    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    // Per-event performance file: one row per event, in event order.
    const std::string perf_path =
        std::string(Backend::output_dir) + "/performance.csv";
    std::ofstream perf_file{perf_path};
    if (!perf_file) {
        throw std::runtime_error("Cannot open " + perf_path);
    }
    perf_file << "backend,event,n_cells,n_measurements,n_spacepoints,n_seeds,"
                 "n_found_tracks,n_fitted_tracks,t_read_ms,t_h2d_ms,"
                 "t_clusterization_ms,t_spacepoints_ms,t_seeding_ms,"
                 "t_params_ms,t_finding_ms,t_fitting_ms,t_wall_ms,t_d2h_ms,"
                 "n_threads,thread,t_start_ms,t_end_ms\n";

    std::uint64_t n_cells = 0, n_measurements = 0, n_spacepoints = 0,
                  n_seeds = 0, n_found_tracks = 0, n_fitted_tracks = 0;
    event_times total;
    for (const event_result& r : results) {
        const event_times& t = r.times;
        perf_file << Backend::name << ',' << r.event << ',' << r.n_cells << ','
                  << r.n_measurements << ',' << r.n_spacepoints << ','
                  << r.n_seeds << ',' << r.n_found_tracks << ','
                  << r.n_fitted_tracks << ',' << t.read << ',' << t.h2d << ','
                  << t.clusterization << ',' << t.spacepoints << ','
                  << t.seeding << ',' << t.params << ',' << t.finding << ','
                  << t.fitting << ',' << t.wall << ',' << t.d2h << ','
                  << n_threads << ',' << r.thread << ',' << r.t_start << ','
                  << r.t_end << '\n';

        n_cells += r.n_cells;
        n_measurements += r.n_measurements;
        n_spacepoints += r.n_spacepoints;
        n_seeds += r.n_seeds;
        n_found_tracks += r.n_found_tracks;
        n_fitted_tracks += r.n_fitted_tracks;
        total.read += t.read;
        total.h2d += t.h2d;
        total.clusterization += t.clusterization;
        total.spacepoints += t.spacepoints;
        total.seeding += t.seeding;
        total.params += t.params;
        total.finding += t.finding;
        total.fitting += t.fitting;
        total.wall += t.wall;
        total.d2h += t.d2h;
    }

    TRACCC_INFO("==> Statistics (" << Backend::name << ") ... ");
    TRACCC_INFO("- read    " << n_cells << " cells");
    TRACCC_INFO("- created " << n_measurements << " measurements");
    TRACCC_INFO("- created " << n_spacepoints << " spacepoints");
    TRACCC_INFO("- created " << n_seeds << " seeds");
    TRACCC_INFO("- found   " << n_found_tracks << " tracks");
    TRACCC_INFO("- fitted  " << n_fitted_tracks << " tracks");
    TRACCC_INFO("==> Times summed over the events (ms):");
    TRACCC_INFO("  File reading         " << total.read);
    TRACCC_INFO("  H2D copy             " << total.h2d);
    TRACCC_INFO("  Clusterization       " << total.clusterization);
    TRACCC_INFO("  Spacepoint formation " << total.spacepoints);
    TRACCC_INFO("  Seeding              " << total.seeding);
    TRACCC_INFO("  Track params         " << total.params);
    TRACCC_INFO("  Track finding        " << total.finding);
    TRACCC_INFO("  Track fitting        " << total.fitting);
    TRACCC_INFO("  Wall time            " << total.wall);
    TRACCC_INFO("  D2H copy             " << total.d2h);
    TRACCC_INFO("==> Event loop: " << loop_ms << " ms for " << events.size()
                                   << " events with " << n_threads
                                   << " thread(s) -> "
                                   << (loop_ms > 0. ? 1000. * events.size() / loop_ms
                                                    : 0.)
                                   << " events/s");

    return 0;
}

/**
 * @brief Removes --threads=N (or --threads N) from the command line, because
 *        the traccc option parser would refuse it, and returns N (default 1).
 */
inline unsigned int extract_threads(int& argc, char* argv[]) {
    unsigned int n_threads = 1;
    const auto parse = [](std::string_view value) {
        const unsigned long n = std::stoul(std::string(value));
        if (n == 0) {
            throw std::invalid_argument("--threads must be at least 1");
        }
        return static_cast<unsigned int>(n);
    };
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.rfind("--threads=", 0) == 0) {
            n_threads = parse(arg.substr(std::string_view("--threads=").size()));
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = parse(argv[++i]);
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    argv[argc] = nullptr;
    return n_threads;
}

/**
 * @brief Parses the command line options and runs the chain with @p Backend.
 */
template <typename Backend>
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        std::string("TracccReco_") + Backend::name, traccc::Logging::Level::INFO);

    const unsigned int n_threads = extract_threads(argc, argv);

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
        std::string("Full Tracking Chain (") + Backend::name +
            ") [extra option: --threads=N, events processed in parallel]",
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
                        seeding_gbts_opts.useGBTS, n_threads);
}

}  // namespace reco