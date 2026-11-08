/**
 * @file track_csv_writer.hpp
 * @brief Writes the reconstructed tracks of an event to CSV files.
 */
#pragma once

// Project include(s).

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_constituent_link.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/io/utils.hpp"
// System include(s).
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reco {

/**
 * @brief Writes two files in @p directory for event @p event:
 *        - event<N>-tracks.csv: one row per track, with the fit results
 *        - event<N>-track-measurements.csv: one row per measurement of each track
 * @param directory Output folder, e.g. the input folder of the event.
 * @param event Event number.
 * @param tracks Fitted tracks, with their states and measurements.
 */
inline void write_tracks_csv(
    std::string_view directory, std::size_t event,
    traccc::edm::track_container<traccc::default_algebra>::const_view tracks_view) {

  const traccc::edm::track_container<traccc::default_algebra>::const_device tracks{
      tracks_view};

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

  track_file << "track_id,fit_outcome,chi2,ndf,pval,n_measurements,n_holes,"
                "loc0,loc1,phi,theta,qop\n";
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
      } else if (type == traccc::edm::track_constituent_link::track_state) {
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
                << meas.local_position()[0] << ',' << meas.local_position()[1]
                << '\n';
      ++n_measurements;
    }

    // Fitted track parameters, bound to the surface they refer to.
    const auto& params = track.params();
    const auto local = params.bound_local();
    track_file << i << ',' << static_cast<unsigned int>(track.fit_outcome())
               << ',' << track.chi2() << ',' << track.ndf() << ','
               << track.pval() << ',' << n_measurements << ','
               << track.nholes() << ',' << local[0] << ',' << local[1] << ','
               << params.phi() << ',' << params.theta() << ',' << params.qop()
               << '\n';
  }
}

}  // namespace reco