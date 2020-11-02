function(add_sah_kd_tree_test test_file)
  get_filename_component(name "${test_file}" NAME_WE)
  set(test_name "check-sah_kd_tree-${name}")

  add_test(
    NAME "${test_name}"
    COMMAND "builder" "${test_file}")

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

add_sah_kd_tree_test("data/triangle.obj")
add_sah_kd_tree_test("data/aa_triangle.obj")
add_sah_kd_tree_test("data/coincident_triangles.obj")
add_sah_kd_tree_test("data/aa_parallel_non_coincident_triangles.obj")
add_sah_kd_tree_test("data/box.obj")
add_sah_kd_tree_test("data/aa_box.obj")
add_sah_kd_tree_test("data/tetrahedron.obj")
add_sah_kd_tree_test("data/box_inside_box.obj")
