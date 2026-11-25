# Build of all the C++ programs of the project, against an installed traccc
# (see exec/build.sh, which installs it in traccc-install/):
#   simulation/      traccc_toy_mc      toy MC simulation
#   reconstruction/  traccc_reco_gpu    traccc on the GPU (CUDA)
#                    traccc_reco_cpu    traccc on the CPU (Alpaka)
#                    traccc_hough_reco  Hough transform (plain C++)
#
# Usage, from the project folder:
#   cmake -S src -B build -DCMAKE_PREFIX_PATH="$PWD/traccc-install;$HOME/local"
#   cmake --build build -j 16
# All the executables are written to build/bin. The options of the
# reconstruction (RECO_BUILD_GPU, RECO_BUILD_CPU, RECO_DATA_DIR) can be given
# here too. Each subfolder can also be built on its own (cmake -S src/<folder>).
cmake_minimum_required(VERSION 3.25)
project(cmepda_exam LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(simulation)
add_subdirectory(reconstruction)