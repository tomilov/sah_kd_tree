if(NOT ${SAH_KD_TREE_GENERATE_DOCS})
    return()
endif()

find_package(Doxygen REQUIRED)
if(DOXYGEN_DOT_FOUND)
  set(HAS_DOT_VALUE "YES")
else()
  set(HAS_DOT_VALUE "NO")
endif()

configure_file("sah_kd_tree.cfg.in" "sah_kd_tree.cfg" @ONLY)

add_custom_target(
    "docs"
    DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/sah_kd_tree.cfg"
    VERBATIM
    COMMAND
        "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/sah_kd_tree.cfg"
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT
        "Generating API documentation with Doxygen")
