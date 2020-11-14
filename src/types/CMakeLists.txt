add_library("types" INTERFACE)

target_sources(
  "types"
  INTERFACE
    "include/SahKdTree/types.hpp")

target_include_directories(
  "types"
  INTERFACE
    "include")
