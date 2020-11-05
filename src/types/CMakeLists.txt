add_library("types" INTERFACE)

target_sources(
  "types"
  INTERFACE
    "include/sah_kd_tree/types.hpp")

target_include_directories(
  "types"
  INTERFACE
    "include"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
