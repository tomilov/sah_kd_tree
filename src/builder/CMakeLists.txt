project("builder")

add_library("lib${PROJECT_NAME}")
set_property(TARGET "lib${PROJECT_NAME}" PROPERTY LIBRARY_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "build_from_triangles.hpp"
        "build_from_triangles.cu")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        "sah_kd_tree"
        "scene_loader"
    PUBLIC
        Qt5::Core)

add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "main.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "lib${PROJECT_NAME}")

add_subdirectory("tests")
