cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project("sah_kd_tree" LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Utils)

cmake_policy(SET CMP0127 NEW)
include(CMakeDependentOption)

cmake_policy(SET CMP0105 NEW) # link options for device link step

add_subdirectory("src")

option(GENERATE_DOCS "Whether to build the Doxygen documentation" OFF)
add_subdirectory("docs")
