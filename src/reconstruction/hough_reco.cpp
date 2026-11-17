/**
 * @file hough_reco.cpp
 * @brief Track reconstruction with the Hough transform (hough.hpp), on the
 *        spacepoints written by traccc_reco_<cpu|gpu> --write-spacepoints.
 *
 * The program replaces seeding, track finding and track fitting of traccc:
 * clusterization and spacepoint formation are those of traccc, so both
 * reconstructions start from the same spacepoints. For every event it writes
 * the tracks in the same CSV format as traccc_reco (tracks + track
 * measurements), so they can be analysed with the same scripts, and one row of
 * performance.csv with the time of the Hough transform.
 *
 * Usage:
 *   hough_reco <spacepoints folder> [--config config/hough.conf] [--output folder]
 */

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
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Output folder for the CSV files, set by CMake (see CMakeLists.txt).
#ifndef RECO_DATA_DIR
#define RECO_DATA_DIR "data"
#endif

namespace {

using clock_type = std::chrono::steady_clock;

/// Milliseconds elapsed since @p start.
double ms_since(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

/// One spacepoint read from file, with the measurement it comes from.
struct spacepoint_row {
    double x = 0., y = 0., z = 0.;
    std::uint64_t geometry_id = 0;
    double local0 = 0., local1 = 0.;
};

/// Splits one CSV line at the commas.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

/**
 * @brief Reads event<N>-spacepoints.csv.
 * @throws std::runtime_error if the file cannot be read or a column is missing.
 */
std::vector<spacepoint_row> read_spacepoints(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = split(hough::detail::trim(line));
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
        const std::vector<std::string> f = split(hough::detail::trim(line));
        if (f.size() < header.size()) {
            continue;  // empty or cut line
        }
        rows.push_back({std::stod(f[cx]), std::stod(f[cy]), std::stod(f[cz]),
                        static_cast<std::uint64_t>(std::stoull(f[cg])),
                        std::stod(f[c0]), std::stod(f[c1])});
    }
    return rows;
}

/**
 * @brief Writes the tracks of one event in the format of track_csv_writer.hpp,
 *        plus some columns of the Hough fit at the end of tracks.csv.
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

    // No chi2 and p-value: the Hough fit has no measurement errors (ndf = 0)
    track_file << "track_id,fit_outcome,chi2,ndf,pval,n_measurements,n_holes,"
                  "loc0,loc1,phi,theta,qop,pt_gev,z0_mm,rms_xy_mm,rms_z_mm\n";
    meas_file << "track_id,geometry_id,local0,local1\n";
    for (std::size_t t = 0; t < tracks.size(); ++t) {
        const hough::track& tr = tracks[t];
        // The parameters are given at the first hit, as those of traccc
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

}  // namespace

/**
 * @brief Main routine: Hough transform on every event of a spacepoints folder.
 * @param argc Number of command line arguments.
 * @param argv The spacepoints folder, then --config and --output (optional).
 * @return 0 on success, 1 on a wrong command line or an error.
 */
int main(int argc, char* argv[]) {
    std::filesystem::path input, config_path = "config/hough.conf",
                                 output = RECO_DATA_DIR "/reco_hough";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (input.empty() && arg.rfind("--", 0) != 0) {
            input = arg;
        } else {
            input.clear();
            break;
        }
    }
    if (input.empty()) {
        std::cerr << "Usage: " << argv[0] << " <spacepoints folder> "
                  << "[--config config/hough.conf] [--output folder]\n";
        return 1;
    }

    try {
        const hough::config cfg = hough::read_config(config_path.string());

        // Events of the input folder, in order
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

        // Output folder, without the tracks of a previous run
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
        perf << "backend,event,n_spacepoints,n_fitted_tracks,t_read_ms,t_hough_ms,"
                "t_wall_ms\n";

        double total_hough = 0.;
        std::size_t total_tracks = 0;
        for (const auto& [event, path] : events) {
            const auto start = clock_type::now();
            const std::vector<spacepoint_row> rows = read_spacepoints(path);
            const double t_read = ms_since(start);

            std::vector<hough::point> points;
            points.reserve(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                points.push_back({rows[i].x, rows[i].y, rows[i].z, rows[i].geometry_id, i});
            }

            // Only the Hough transform and the fit are timed
            const auto hough_start = clock_type::now();
            const std::vector<hough::track> tracks = hough::find_tracks(points, cfg);
            const double t_hough = ms_since(hough_start);

            write_tracks(output, event, rows, tracks);
            perf << "hough," << event << ',' << rows.size() << ',' << tracks.size() << ','
                 << t_read << ',' << t_hough << ',' << t_read + t_hough << '\n';
            total_hough += t_hough;
            total_tracks += tracks.size();
        }

        std::cout << "Hough transform: " << events.size() << " events, " << total_tracks
                  << " tracks, " << total_hough << " ms ("
                  << total_hough / static_cast<double>(events.size())
                  << " ms/event). Tracks and performance.csv in " << output.string()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}