/**
 * @file spacepoint_csv_writer.hpp
 * @brief Writes the spacepoints of an event to a CSV file, as input of
 *        programs outside traccc (e.g. the Hough transform, hough_reco).
 */
#pragma once

// Project include(s).
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
 * @param directory Output folder (it must exist).
 * @param event Event number, used in the file name.
 * @param spacepoints Spacepoints of the event, in host memory.
 * @param measurements Measurements of the event, in host memory.
 * @throws std::runtime_error if the file cannot be opened.
 */
template <typename spacepoint_collection_t, typename measurement_collection_t>
inline void write_spacepoints_csv(std::string_view directory, std::size_t event,
                                  const spacepoint_collection_t& spacepoints,
                                  const measurement_collection_t& measurements) {
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
             << meas.local_position()[0] << ',' << meas.local_position()[1] << '\n';
    }
}

}  // namespace reco