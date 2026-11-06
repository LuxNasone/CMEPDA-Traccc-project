// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/detector.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/generation.hpp"
#include "traccc/options/output_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/simulation/event_generators.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"

// detray include(s).
#include <detray/test/common/build_toy_detector.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

using namespace traccc;

struct toy_config {
  unsigned int barrel = 4u;
  unsigned int endcap = 7u;
  double bfield_tesla = 2.;
  double smearing_um = 50.;
  std::uint64_t seed = 0u;
  std::vector<std::string> traccc_args;
};

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1u);
}

toy_config read_config(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Impossibile aprire " + path);
  }

  toy_config cfg;
  std::string line;
  while (std::getline(file, line)) {
    line = trim(line.substr(0u, line.find('#')));
    if (line.empty()) {
      continue;
    }

    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("Riga non valida: " + line);
    }
    const std::string key = trim(line.substr(0u, separator));
    const std::string value = trim(line.substr(separator + 1u));

    try {
      if (key == "barrel") {
        cfg.barrel = static_cast<unsigned int>(std::stoul(value));
      } else if (key == "endcap") {
        cfg.endcap = static_cast<unsigned int>(std::stoul(value));
      } else if (key == "bfield-tesla") {
        cfg.bfield_tesla = std::stod(value);
      } else if (key == "smearing-um") {
        cfg.smearing_um = std::stod(value);
      } else if (key == "seed") {
        cfg.seed = static_cast<std::uint64_t>(std::stoull(value));
      } else {
        cfg.traccc_args.push_back("--" + key + "=" + value);
      }
    } catch (const std::exception&) {
      throw std::runtime_error("Valore non valido per " + key + ": " + value);
    }
  }
  return cfg;
}

int simulate(const toy_config& cfg,
             const traccc::opts::generation& generation_opts,
             const traccc::opts::output_data& output_opts,
             const traccc::opts::track_propagation& propagation_opts) {
  // Use deterministic random number generator for testing
  using uniform_gen_t =
      detray::detail::random_numbers<scalar,
                                     std::uniform_real_distribution<scalar>>;

  // Memory resource
  vecmem::host_memory_resource host_mr;

  /*****************************
   * Build a toy geometry
   *****************************/

  // Detector type
  using detector_type = detray::detector_host_t<traccc::toy_detector>;

  // B field value and its type
  // @TODO: Set B field as argument
  using b_field_t = covfie::field<traccc::const_bfield_backend_t<scalar>>;
  const vector3 B{0.f, 0.f, static_cast<scalar>(cfg.bfield_tesla) * traccc::unit<scalar>::T};
  b_field_t field = traccc::construct_const_bfield(B)
                        .as_field<traccc::const_bfield_backend_t<scalar>>();

  // Create the toy geometry
  detray::toy_det_config<scalar> toy_cfg{};
  toy_cfg.n_brl_layers(cfg.barrel).n_edc_layers(cfg.endcap);
  // @TODO: Increase the material budget again
  toy_cfg.module_mat_thickness(0.11f * traccc::unit<scalar>::mm);
  const auto [det, name_map] =
      detray::build_toy_detector<traccc::default_algebra>(host_mr, toy_cfg);

  /***************************
   * Generate simulation data
   ***************************/

  // Origin of particles
  using generator_type =
      detray::random_track_generator<traccc::free_track_parameters<>,
                                     uniform_gen_t>;
  generator_type::configuration gen_cfg{};
  gen_cfg.n_tracks(generation_opts.gen_nparticles);
  gen_cfg.origin(traccc::point3{generation_opts.vertex[0],
                                generation_opts.vertex[1],
                                generation_opts.vertex[2]});
  gen_cfg.origin_stddev(traccc::point3{generation_opts.vertex_stddev[0],
                                       generation_opts.vertex_stddev[1],
                                       generation_opts.vertex_stddev[2]});
  gen_cfg.phi_range(generation_opts.phi_range);
  gen_cfg.theta_range(generation_opts.theta_range);
  gen_cfg.mom_range(generation_opts.mom_range);

  if (cfg.seed != 0u) {
    gen_cfg.seed(cfg.seed);
  }

  generator_type generator(gen_cfg);

  // Smearing value for measurements
  const scalar smearing = static_cast<scalar>(cfg.smearing_um) * traccc::unit<scalar>::um;
  traccc::measurement_smearer<traccc::default_algebra> meas_smearer(smearing, smearing);

  // Type declarations
  using writer_type = traccc::smearing_writer<
      traccc::measurement_smearer<traccc::default_algebra>>;

  // Writer config
  typename writer_type::config smearer_writer_cfg{meas_smearer};

  // Run simulator
  const std::string full_path = io::data_directory() + output_opts.directory;

  boost::filesystem::create_directories(full_path);

  auto sim =
      traccc::simulator<detector_type, b_field_t, generator_type, writer_type>(
          generation_opts.ptc_type, generation_opts.events, det, field,
          std::move(generator), std::move(smearer_writer_cfg), full_path);
  sim.get_config().propagation = propagation_opts;

  sim.get_config().seed_offset = cfg.seed * 1'000'000u;

  sim.run();

  // Create detector file
  auto writer_cfg = detray::io::detector_writer_config{}
                        .format(detray::io::format::json)
                        .replace_files(true);
  detray::io::write_detector(det, name_map, writer_cfg);

  return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Uso: " << argv[0] << " <file di configurazione>\n";
    return 1;
  }

  std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
      "TracccExampleSimulateToyDetector", traccc::Logging::Level::INFO);

  try {
    const toy_config cfg = read_config(argv[1]);

    std::vector<std::string> args{argv[0]};
    args.insert(args.end(), cfg.traccc_args.begin(), cfg.traccc_args.end());

    std::vector<char*> c_args;
    for (std::string& arg : args) {
      c_args.push_back(arg.data());
    }

    traccc::opts::generation generation_opts;
    traccc::opts::output_data output_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::program_options program_opts{
        "Toy-Detector Simulation",
        {generation_opts, output_opts, propagation_opts},
        static_cast<int>(c_args.size()),
        c_args.data(),
        logger->cloneWithSuffix("Options")};

    return simulate(cfg, generation_opts, output_opts, propagation_opts);
  } catch (const std::exception& error) {
    std::cerr << "Errore: " << error.what() << '\n';
    return 1;
  }
}
