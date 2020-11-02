ProcessorCount(N)
if(N EQUAL 0)
  set(N 1)
endif()

add_custom_target(
  "check"
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
    --output-on-failure
    --parallel ${N}
  DEPENDS
    "builder")

add_subdirectory("sah_kd_tree")

