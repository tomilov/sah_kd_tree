ProcessorCount(N)

add_custom_target(
  "check"
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
    --output-on-failure
    --parallel ${N}
  DEPENDS
    "Builder")

add_subdirectory("SahKdTree")

