cmake_minimum_required( VERSION 3.28.3 )
project( CheckSYCL LANGUAGES CXX )
list( INSERT CMAKE_MODULE_PATH 0       "/home/local/ADUNIPI/l.nasone/Exam_2/traccc-install/lib/cmake/vecmem-1.28.0"       "/home/local/ADUNIPI/l.nasone/Exam_2/traccc-install/lib/cmake/vecmem-1.28.0/hip"       "/home/local/ADUNIPI/l.nasone/Exam_2/traccc-install/lib/cmake/vecmem-1.28.0/sycl" )
enable_language( SYCL )
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
   "set( CMAKE_SYCL_COMPILER \"${CMAKE_SYCL_COMPILER}\" )" )