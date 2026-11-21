/**
 * @file csv_writer.hpp
 * @brief Writes the spacepoints and the reconstructed tracks of an event
 *        to CSV files.
 */
#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/track_constituent_link.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/io/utils.hpp"

// System include(s).
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reco {

/**
 * @brief Writes event<N>-spacepoints.csv in @p directory: one row per
 *        spacepoint, with its global position and the module and local
 *        position of the measurement it comes from.
 * @param directory Output folder (created if missing).
 * @param event Event number, used in the file name.
 * @param spacepoints Spacepoints of the event, in host memory.
 * @param measurements Measurements of the event, in host memory.
 * @throws std::runtime_error if the file cannot be opened.
 */
template <typename spacepoint_collection_t, typename measurement_collection_t>
inline void write_spacepoints_csv(std::string_view directory, std::size_t event,
                                  const spacepoint_collection_t& spacepoints,
                                  const measurement_collection_t& measurements) {
    std::filesystem::create_directories(directory);
    const std::string path =
        (std::filesystem::path(directory) /
         traccc::io::get_event_filename(event, "-spacepoints.csv"))
            .string();

    std::ofstream file{path};
    if (!file) {
        throw std::runtime_error("Cannot open " + path);
    }
    file << std::setprecision(9);

    file << "spacepoint_id,x,y,z,measurement_id,geometry_id,local0,local1\n";
    for (std::size_t i = 0; i < spacepoints.size(); ++i) {
        const auto sp = spacepoints.at(i);
        const auto meas_index = sp.measurement_index_1();
        const auto meas = measurements.at(meas_index);
        file << i << ',' << sp.x() << ',' << sp.y() << ',' << sp.z() << ','
             << meas_index << ',' << meas.surface_link().value() << ','
             << meas.local_position()[0] << ',' << meas.local_position()[1]
             << '\n';
    }
}

/**
 * @brief Writes two files in @p directory for event @p event:
 *        - event<N>-tracks.csv: one row per track, with the fit results: the
 *          parameters on the surface of the first measurement and their
 *          covariance (the 15 elements of the upper triangle of the 5x5
 *          matrix of loc0, loc1, phi, theta, qop, as cov_<a>_<b>)
 *        - event<N>-track-measurements.csv: one row per measurement of each
 *          track
 * @param directory Output folder (created if missing).
 * @param event Event number, used in the file names.
 * @param tracks_view Fitted tracks, with their states and measurements.
 * @throws std::runtime_error if a file cannot be opened or a track has an
 *         unknown constituent type.
 */
inline void write_tracks_csv(
    std::string_view directory, std::size_t event,
    traccc::edm::track_container<traccc::default_algebra>::const_view
        tracks_view) {
    const traccc::edm::track_container<traccc::default_algebra>::const_device
        tracks{tracks_view};

    std::filesystem::create_directories(directory);
    const auto path = [&](std::string_view suffix) {
        return (std::filesystem::path(directory) /
                traccc::io::get_event_filename(event, suffix))
            .string();
    };

    std::ofstream track_file{path("-tracks.csv")};
    std::ofstream meas_file{path("-track-measurements.csv")};
    if (!track_file || !meas_file) {
        throw std::runtime_error("Cannot open the track CSV files in " +
                                 std::string(directory));
    }
    track_file << std::setprecision(9);
    meas_file << std::setprecision(9);

    // Parameters of the covariance written to file, in the order of the
    // bound parameters of traccc (the time, e_bound_time, is not used)
    constexpr std::size_t n_cov = 5;
    constexpr const char* cov_names[n_cov] = {"loc0", "loc1", "phi", "theta", "qop"};
    constexpr unsigned int cov_index[n_cov] = {
        traccc::e_bound_loc0, traccc::e_bound_loc1, traccc::e_bound_phi,
        traccc::e_bound_theta, traccc::e_bound_qoverp};

    track_file << "track_id,fit_outcome,chi2,ndf,pval,n_measurements,n_holes,"
                  "loc0,loc1,phi,theta,qop";
    for (std::size_t a = 0; a < n_cov; ++a) {
        for (std::size_t b = a; b < n_cov; ++b) {
            track_file << ",cov_" << cov_names[a] << '_' << cov_names[b];
        }
    }
    track_file << '\n';
    meas_file << "track_id,geometry_id,local0,local1\n";

    using size_type = decltype(tracks.tracks.size());
    for (size_type i = 0; i < tracks.tracks.size(); ++i) {
        const auto track = tracks.tracks.at(i);

        // Measurements of the track: follow each link to its measurement.
        unsigned int n_measurements = 0;
        for (const auto& [type, idx] : track.constituent_links()) {
            unsigned int meas_index = 0;
            if (type == traccc::edm::track_constituent_link::measurement) {
                meas_index = idx;
            } else if (type ==
                       traccc::edm::track_constituent_link::track_state) {
                const auto state = tracks.states.at(idx);
                if (state.is_hole()) {
                    continue;  // surface crossed without a measurement
                }
                meas_index = state.measurement_index();
            } else {
                throw std::runtime_error("Unknown track constituent type");
            }

            const auto meas = tracks.measurements.at(meas_index);
            meas_file << i << ',' << meas.surface_link().value() << ','
                      << meas.local_position()[0] << ','
                      << meas.local_position()[1] << '\n';
            ++n_measurements;
        }

        // Fitted track parameters, bound to the surface they refer to.
        const auto& params = track.params();
        const auto local = params.bound_local();
        track_file << i << ',' << static_cast<unsigned int>(track.fit_outcome())
                   << ',' << track.chi2() << ',' << track.ndf() << ','
                   << track.pval() << ',' << n_measurements << ','
                   << track.nholes() << ',' << local[0] << ',' << local[1]
                   << ',' << params.phi() << ',' << params.theta() << ','
                   << params.qop();
        const auto& cov = params.covariance();
        for (std::size_t a = 0; a < n_cov; ++a) {
            for (std::size_t b = a; b < n_cov; ++b) {
                track_file << ','
                           << traccc::getter::element(cov, cov_index[a], cov_index[b]);
            }
        }
        track_file << '\n';
    }
}

}  // namespace reco