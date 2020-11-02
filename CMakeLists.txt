cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project("sah_kd_tree" LANGUAGES CXX CUDA)

list(
  APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake"
  "${PROJECT_SOURCE_DIR}/cmake/modules")

include(utils)

include(CTest)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(GENERATE_DOCS "Whether or not to build the Doxygen documentation" OFF)

add_subdirectory(docs)
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

