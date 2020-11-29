function(add_sah_kd_tree_test test_file)
  get_filename_component(name "${test_file}" NAME_WE)
  set(test_name "check-SahKdTree-${name}")

  add_test(
    NAME "${test_name}"
    COMMAND
      "Builder"
      "${CMAKE_CURRENT_SOURCE_DIR}/data/${test_file}")

  if(ARGC GREATER 1)
    set_tests_properties(
      "${test_name}"
      PROPERTIES
        RUN_SERIAL "${ARGV1}")
  endif()
  if(ARGC GREATER 2)
    set_tests_properties(
      "${test_name}"
      PROPERTIES
        TIMEOUT "${ARGV2}")
  endif()
endfunction()

add_sah_kd_tree_test("triangle.obj")
add_sah_kd_tree_test("aa_triangle.obj")
add_sah_kd_tree_test("coincident_triangles.obj")
add_sah_kd_tree_test("aa_parallel_non_coincident_triangles.obj")
add_sah_kd_tree_test("box.obj")
add_sah_kd_tree_test("aa_box.obj")
add_sah_kd_tree_test("tetrahedron.obj")
add_sah_kd_tree_test("box_inside_box.obj")
