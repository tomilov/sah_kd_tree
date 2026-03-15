cmake_minimum_required(VERSION 4.2.3..4.2.3 FATAL_ERROR)
project(
    "sah_kd_tree"
    LANGUAGES
        CXX CUDA
    VERSION
        0.0.1.0
)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Utils)

include(CMakeDependentOption)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("external" EXCLUDE_FROM_ALL SYSTEM)

option(SAH_KD_TREE_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if(SAH_KD_TREE_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY "clang-tidy" REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}" --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy --header-filter=${CMAKE_SOURCE_DIR}/include/.*)
endif()

option(SAH_KD_TREE_ENABLE_CPPCHECK "Enable cppcheck" OFF)
if(SAH_KD_TREE_ENABLE_CPPCHECK)
    find_program(CPPCHECK "cppcheck" REQUIRED)
    set(CMAKE_CXX_CPPCHECK "${CPPCHECK}" --enable=all --suppress=missingInclude)
endif()

option(SAH_KD_TREE_ENABLE_IWYU "Enable include-what-you-use" OFF)
if(SAH_KD_TREE_ENABLE_IWYU)
  find_program(IWYU_EXE NAMES "include-what-you-use" REQUIRED)
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
endif()

add_subdirectory("src")
