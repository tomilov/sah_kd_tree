cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(
    "sah_kd_tree"
    LANGUAGES
        CXX CUDA
    VERSION
        0.0.1.0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Utils)

cmake_policy(SET CMP0054 NEW)

cmake_policy(SET CMP0127 NEW)
include(CMakeDependentOption)

cmake_policy(SET CMP0105 NEW) # link options for device link step

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("external" EXCLUDE_FROM_ALL SYSTEM)
add_subdirectory("src")

option(SAH_KD_TREE_GENERATE_DOCS "Whether to build the Doxygen documentation" OFF)
add_subdirectory("docs")
