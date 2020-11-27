find_package(Doxygen REQUIRED)
if(DOXYGEN_DOT_FOUND)
  set(HAS_DOT_VALUE "YES")
else()
  set(HAS_DOT_VALUE "NO")
endif()

configure_file("SahKdTree.cfg.in" "SahKdTree.cfg" @ONLY)

add_custom_target(
  "docs"
  COMMAND
    "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/SahKdTree.cfg"
  WORKING_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT
    "Generating API documentation with Doxygen"
  VERBATIM)
