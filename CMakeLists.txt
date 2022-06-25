cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project("sah_kd_tree")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Utils)

add_subdirectory("src")

option(GENERATE_DOCS "Whether to build the Doxygen documentation" OFF)
add_subdirectory("docs")
