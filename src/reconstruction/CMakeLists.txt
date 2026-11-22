cmake_minimum_required(VERSION 3.25)
project(traccc_performance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(traccc REQUIRED)

set(RECO_DATA_DIR "${CMAKE_SOURCE_DIR}/../../results" CACHE PATH
    "Folder where the reconstruction writes its output")

add_executable(traccc_reco_gpu traccc/reco_gpu.cpp)
target_link_libraries(traccc_reco_gpu PRIVATE
    traccc::core traccc::io traccc::cuda traccc::options vecmem::cuda)
target_compile_definitions(traccc_reco_gpu PRIVATE
    RECO_DATA_DIR="${RECO_DATA_DIR}")

add_executable(traccc_hough_reco hough/hough_reco.cpp)
target_link_libraries(traccc_hough_reco PRIVATE traccc::core traccc::io)