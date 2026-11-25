# Build of the toy MC simulation of the project, against an installed traccc
# (cmake --install of traccc-build, see build.sh):
#   traccc_toy_mc   toy detector simulation (detray toy geometry, traccc
#                   simulator with the seed offset of seeded_simulator.hpp)
#
# Usage, from the project folder:
#   cmake -S src/simulation -B sim-build \
#         -DCMAKE_PREFIX_PATH="$PWD/traccc-install;$HOME/local"
#   cmake --build sim-build -j 16
# The executable is written to sim-build/bin.
cmake_minimum_required(VERSION 3.25)
project(cmepda_simulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(traccc REQUIRED)
find_package(detray REQUIRED)
find_package(covfie REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

# Run path with the lib folders of traccc and Boost, also for the libraries
# that traccc loads indirectly (see src/reconstruction/CMakeLists.txt)
get_filename_component(traccc_lib_dir "${traccc_DIR}/../.." ABSOLUTE)
list(APPEND CMAKE_BUILD_RPATH "${traccc_lib_dir}")
if(Boost_DIR)
    get_filename_component(boost_lib_dir "${Boost_DIR}/../.." ABSOLUTE)
    list(APPEND CMAKE_BUILD_RPATH "${boost_lib_dir}")
endif()

add_executable(traccc_toy_mc ToyMC.cpp)
target_link_libraries(traccc_toy_mc PRIVATE
    vecmem::core
    traccc::io
    traccc::core
    traccc::options
    traccc::simulation
    detray::io
    detray::test_common
    covfie::core
    Boost::filesystem)
target_link_options(traccc_toy_mc PRIVATE "LINKER:--disable-new-dtags")

