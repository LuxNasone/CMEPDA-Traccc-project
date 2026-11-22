/**
 * @file hough_reco.cpp
 * @brief Track reconstruction with the Hough transform (hough.hpp), on the
 *        spacepoints written by traccc_reco_<cpu|gpu> --write-spacepoints.
 *
 * The program replaces seeding, track finding and track fitting of traccc:
 * clusterization and spacepoint formation are those of traccc, so both
 * reconstructions start from the same spacepoints. For every event it writes
 * the tracks in the same CSV format as traccc_reco (tracks + track
 * measurements, see csv_writer.hpp), so they can be analysed with the same
 * scripts, and one row of performance.csv with the time of the Hough
 * transform.
 *
 * The events are independent, so with --threads=N they are shared among N
 * threads: each thread takes the next event not yet processed, as in
 * traccc_reco (reco.hpp). The algorithm itself runs on one thread per event.
 *
 * Usage:
 *   hough_reco <spacepoints folder> [--config config/hough.conf]
 *              [--output folder] [--threads N]
 */

// Project include(s).
#include "../reconstruction/utils.hpp"
#include "hough.hpp"

// System include(s).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Data folder of the project, set by CMake (see CMakeLists.txt). The default
// output folder is RECO_DATA_DIR/reco_hough.
#ifndef RECO_DATA_DIR
#define RECO_DATA_DIR "data"
#endif

namespace {

using clock_type = utils::clock_type;

/// One row of event<N>-spacepoints.csv: global position of the spacepoint,
/// module and local position of the measurement it comes from.
struct spacepoint_row {
    double x = 0., y = 0., z = 0.;
    std::uint64_t geometry_id = 0;
    double local0 = 0., local1 = 0.;
};

/**
 * @brief Reads event<N>-spacepoints.csv. The columns are found by name in the
 *        header, so their order does not matter.
 * @param path File to read.
 * @return One row per spacepoint, in the order of the file.
 * @throws std::runtime_error if the file cannot be read or a column is missing.
 */
std::vector<spacepoint_row> read_spacepoints(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = utils::split(utils::trim(line));
    const auto column = [&](const std::string& name) {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) {
                return i;
            }
        }
        throw std::runtime_error(path.string() + ": no column " + name);
    };
    const std::size_t cx = column("x"), cy = column("y"), cz = column("z"),
                      cg = column("geometry_id"), c0 = column("local0"),
                      c1 = column("local1");

    std::vector<spacepoint_row> rows;
    while (std::getline(file, line)) {
        const std::vector<std::string> f = utils::split(utils::trim(line));
        if (f.size() < header.size()) {
            continue;  // empty or truncated line
        }
        rows.push_back({std::stod(f[cx]), std::stod(f[cy]), std::stod(f[cz]),
                        static_cast<std::uint64_t>(std::stoull(f[cg])),
                        std::stod(f[c0]), std::stod(f[c1])});
    }
    return rows;
}

/**
 * @brief Writes the tracks of one event in the format of csv_writer.hpp,
 *        plus some columns of the Hough fit at the end of tracks.csv.
 * @param folder Output folder (it must exist).
 * @param event Event number, used in the file names.
 * @param rows Spacepoints of the event, as read by read_spacepoints().
 * @param tracks Tracks found by hough::find_tracks(); their hits are indices
 *        into @p rows.
 * @throws std::runtime_error if a file cannot be opened.
 */
void write_tracks(const std::filesystem::path& folder, std::size_t event,
                  const std::vector<spacepoint_row>& rows,
                  const std::vector<hough::track>& tracks) {
    char name[64];
    std::snprintf(name, sizeof(name), "event%09zu", event);
    std::ofstream track_file(folder / (std::string(name) + "-tracks.csv"));
    std::ofstream meas_file(folder / (std::string(name) + "-track-measurements.csv"));
    if (!track_file || !meas_file) {
        throw std::runtime_error("Cannot open the track files in " + folder.string());
    }
    track_file << std::setprecision(9);
    meas_file << std::setprecision(9);

    // The Hough fit has no measurement errors: chi2 and pval are nan, ndf = 0.
    // There are no holes either: only the hits on the track are known.
    track_file << "track_id,fit_outcome,chi2,ndf,pval,n_measurements,n_holes,"
                  "loc0,loc1,phi,theta,qop,pt_gev,z0_mm,rms_xy_mm,rms_z_mm\n";
    meas_file << "track_id,geometry_id,local0,local1\n";
    for (std::size_t t = 0; t < tracks.size(); ++t) {
        const hough::track& tr = tracks[t];
        // loc0, loc1, phi, theta, qop at the first hit, as for traccc
        const spacepoint_row& first = rows[tr.hits.front()];
        track_file << t << ",1,nan,0,nan," << tr.hits.size() << ",0," << first.local0
                   << ',' << first.local1 << ',' << tr.phi << ',' << tr.theta << ','
                   << tr.qop << ',' << tr.pt_gev << ',' << tr.z0_mm << ','
                   << tr.rms_xy_mm << ',' << tr.rms_z_mm << '\n';
        for (std::size_t h : tr.hits) {
            meas_file << t << ',' << rows[h].geometry_id << ',' << rows[h].local0 << ','
                      << rows[h].local1 << '\n';
        }
    }
}

/// Result of one event, written to performance.csv once all threads are done.
struct event_result {
    std::size_t event = 0;
    unsigned int thread = 0;
    std::size_t n_spacepoints = 0;
    std::size_t n_tracks = 0;
    double t_read_ms = 0.;   ///< reading of the spacepoint file
    double t_hough_ms = 0.;  ///< Hough transform and track fit
    double t_start_ms = 0.;  ///< start of the event, from the start of the event loop
    double t_end_ms = 0.;    ///< end of the event (tracks written), same origin
};

/**
 * @brief Processes one event: reads its spacepoints, finds the tracks and
 *        writes them. Each event has its own files, so several events can be
 *        processed at the same time.
 * @param event Event number.
 * @param path Spacepoint file of the event.
 * @param output Output folder of the tracks.
 * @param cfg Configuration of the Hough transform.
 * @param thread Index of the thread that processes the event.
 * @param loop_start Start of the event loop, origin of t_start_ms and t_end_ms.
 * @return Sizes and times of the event.
 * @throws std::runtime_error if a file cannot be read or written.
 */
event_result process_event(std::size_t event, const std::filesystem::path& path,
                           const std::filesystem::path& output, const hough::config& cfg,
                           unsigned int thread, clock_type::time_point loop_start) {
    event_result res;
    res.event = event;
    res.thread = thread;
    res.t_start_ms = utils::ms_since(loop_start);

    const auto start = clock_type::now();
    const std::vector<spacepoint_row> rows = read_spacepoints(path);
    res.t_read_ms = utils::ms_since(start);

    // Input of the Hough transform; index = row of the spacepoint file
    std::vector<hough::point> points;
    points.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        points.push_back({rows[i].x, rows[i].y, rows[i].z, rows[i].geometry_id, i});
    }

    // Only the Hough transform and the track fit are timed, not the I/O
    const auto hough_start = clock_type::now();
    const std::vector<hough::track> tracks = hough::find_tracks(points, cfg);
    res.t_hough_ms = utils::ms_since(hough_start);

    write_tracks(output, event, rows, tracks);
    res.n_spacepoints = rows.size();
    res.n_tracks = tracks.size();
    res.t_end_ms = utils::ms_since(loop_start);
    return res;
}

}  // namespace

/**
 * @brief Main routine: Hough transform on every event of a spacepoints folder.
 * @param argc Number of command line arguments.
 * @param argv The spacepoints folder, then --config, --output and --threads
 *        (optional).
 * @return 0 on success, 1 on a wrong command line or an error.
 */
int main(int argc, char* argv[]) {
    std::filesystem::path input, config_path = "config/hough.conf",
                                 output = RECO_DATA_DIR "/reco_hough";
    std::string threads_arg = "1";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads_arg = argv[++i];
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads_arg = arg.substr(std::string("--threads=").size());
        } else if (input.empty() && arg.rfind("--", 0) != 0) {
            input = arg;
        } else {
            input.clear();  // unknown option or second folder: print the usage
            break;
        }
    }
    if (input.empty()) {
        std::cerr << "Usage: " << argv[0] << " <spacepoints folder> "
                  << "[--config config/hough.conf] [--output folder] [--threads N]\n";
        return 1;
    }

    try {
        const unsigned int n_threads = utils::to_unsigned("--threads", threads_arg);
        if (n_threads == 0) {
            throw std::runtime_error("--threads must be at least 1");
        }
        const hough::config cfg = hough::read_config(config_path.string());

        // Spacepoint files of the input folder, sorted by event number
        std::map<std::size_t, std::filesystem::path> events;
        const std::regex pattern(R"(event(\d+)-spacepoints\.csv)");
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            std::smatch match;
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, match, pattern)) {
                events[std::stoul(match[1].str())] = entry.path();
            }
        }
        if (events.empty()) {
            throw std::runtime_error("No event*-spacepoints.csv in " + input.string() +
                                     " (run traccc_reco_<cpu|gpu> with --write-spacepoints)");
        }

        // Output folder, without the track files of a previous run
        std::filesystem::create_directories(output);
        for (const auto& entry : std::filesystem::directory_iterator(output)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("event", 0) == 0 &&
                name.find("-track") != std::string::npos) {
                std::filesystem::remove(entry.path());
            }
        }

        std::ofstream perf(output / "performance.csv");
        if (!perf) {
            throw std::runtime_error("Cannot open " + (output / "performance.csv").string());
        }
        perf << "backend,event,thread,n_spacepoints,n_fitted_tracks,t_read_ms,"
                "t_hough_ms,t_wall_ms,t_start_ms,t_end_ms\n";

        // Event loop: every thread takes the next event not yet processed.
        // Each result goes to its own slot, so the threads never write the
        // same memory; the first error stops all of them.
        const std::vector<std::pair<std::size_t, std::filesystem::path>> todo(
            events.begin(), events.end());
        std::vector<event_result> results(todo.size());
        std::atomic<std::size_t> next{0};
        std::atomic<bool> stop{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        const unsigned int used_threads =
            static_cast<unsigned int>(std::min<std::size_t>(n_threads, todo.size()));
        const auto loop_start = clock_type::now();
        std::vector<std::thread> workers;
        for (unsigned int t = 0; t < used_threads; ++t) {
            workers.emplace_back([&, t] {
                try {
                    for (std::size_t i = next++; i < todo.size() && !stop; i = next++) {
                        results[i] = process_event(todo[i].first, todo[i].second, output,
                                                   cfg, t, loop_start);
                    }
                } catch (...) {
                    const std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                    stop = true;
                }
            });
        }
        for (std::thread& w : workers) {
            w.join();
        }
        const double loop_ms = utils::ms_since(loop_start);
        if (first_error) {
            std::rethrow_exception(first_error);
        }

        double total_hough = 0.;
        std::size_t total_tracks = 0;
        for (const event_result& r : results) {
            perf << "hough," << r.event << ',' << r.thread << ',' << r.n_spacepoints << ','
                 << r.n_tracks << ',' << r.t_read_ms << ',' << r.t_hough_ms << ','
                 << r.t_end_ms - r.t_start_ms << ',' << r.t_start_ms << ',' << r.t_end_ms
                 << '\n';
            total_hough += r.t_hough_ms;
            total_tracks += r.n_tracks;
        }

        const double n_events = static_cast<double>(results.size());
        std::cout << "Hough transform: " << results.size() << " events, " << total_tracks
                  << " tracks, " << total_hough / n_events << " ms/event (Hough only). "
                  << "Event loop: " << loop_ms << " ms with " << used_threads
                  << " thread(s) -> " << 1000. * n_events / loop_ms << " events/s.\n"
                  << "Tracks and performance.csv in " << output.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}