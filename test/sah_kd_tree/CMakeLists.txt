function(add_sah_kd_tree_test test_file)
  get_filename_component(name "${test_file}" NAME_WE)
  set(test_name "check-sah_kd_tree-${name}")

  add_test(
    NAME "${test_name}"
    COMMAND "builder" "data/${test_file}")

  set_tests_properties(
    "${test_name}"
    PROPERTIES
      TIMEOUT 15)
endfunction()

add_sah_kd_tree_test("triangle.obj")
add_sah_kd_tree_test("aa_triangle.obj")
add_sah_kd_tree_test("coincident_triangles.obj")
add_sah_kd_tree_test("aa_parallel_non_coincident_triangles.obj")
add_sah_kd_tree_test("box.obj")
add_sah_kd_tree_test("aa_box.obj")
add_sah_kd_tree_test("tetrahedron.obj")
add_sah_kd_tree_test("box_inside_box.obj")
