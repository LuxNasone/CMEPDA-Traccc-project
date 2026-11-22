# Build of the reconstruction programs of the project, against an installed
# traccc (cmake --install of traccc-build, see build.sh):
#   traccc_reco_gpu    traccc on the GPU (CUDA)
#   traccc_reco_cpu    traccc on the CPU (Alpaka)
#   traccc_hough_reco  Hough transform on the CPU (plain C++, no traccc)
#
# Usage, from the project folder:
#   cmake -S src/reconstruction -B reco-build \
#         -DCMAKE_PREFIX_PATH=$PWD/traccc-install
#   cmake --build reco-build -j 16
# The executables are written to reco-build/bin.
#
# Options:
#   RECO_BUILD_GPU  build traccc_reco_gpu (default ON)
#   RECO_BUILD_CPU  build traccc_reco_cpu (default ON)
#   RECO_DATA_DIR   folder where the programs write their output
#                   (default: the results folder of the project)
cmake_minimum_required(VERSION 3.25)
project(cmepda_reconstruction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# traccc is built in Release: the timings must be measured with optimizations
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(RECO_BUILD_GPU "Build traccc_reco_gpu (traccc on the GPU, CUDA)" ON)
option(RECO_BUILD_CPU "Build traccc_reco_cpu (traccc on the CPU, Alpaka)" ON)

get_filename_component(default_data_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../results"
                       ABSOLUTE)
set(RECO_DATA_DIR "${default_data_dir}" CACHE PATH
    "Folder where the reconstruction programs write their output")

# Settings shared by every program: the sources of this folder can include
# each other from the top (e.g. "utils.hpp", "csv_writer.hpp"), and the
# output folder is fixed at compile time
function(reco_setup target)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${target} PRIVATE RECO_DATA_DIR="${RECO_DATA_DIR}")
endfunction()

# --- Hough transform: plain C++, multi-threaded --------------------------------
find_package(Threads REQUIRED)
add_executable(traccc_hough_reco hough/hough_reco.cpp)
target_link_libraries(traccc_hough_reco PRIVATE Threads::Threads)
reco_setup(traccc_hough_reco)

# --- traccc reconstruction -----------------------------------------------------
if(RECO_BUILD_GPU OR RECO_BUILD_CPU)
    find_package(traccc REQUIRED)
    find_package(detray REQUIRED)

    # Libraries used by both backends (the same as the traccc examples)
    set(traccc_common_libs
        vecmem::core
        traccc::core
        traccc::io
        traccc::performance
        traccc::device_common
        traccc::options
        traccc::examples_common
        detray::io
        detray::detectors)
endif()

if(RECO_BUILD_GPU)
    add_executable(traccc_reco_gpu traccc/reco_gpu.cpp)
    target_link_libraries(traccc_reco_gpu PRIVATE
        ${traccc_common_libs} vecmem::cuda traccc::cuda traccc::cuda_utils)
    reco_setup(traccc_reco_gpu)
endif()

if(RECO_BUILD_CPU)
    find_package(alpaka REQUIRED)
    add_executable(traccc_reco_cpu traccc/reco_cpu.cpp)
    target_link_libraries(traccc_reco_cpu PRIVATE ${traccc_common_libs} traccc::alpaka)
    reco_setup(traccc_reco_cpu)
endif()

message(STATUS "Reconstruction output folder (RECO_DATA_DIR): ${RECO_DATA_DIR}")