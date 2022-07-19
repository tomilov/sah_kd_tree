project("builder")

add_library("lib${PROJECT_NAME}")
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "build_from_triangles.hpp"
        "build_from_triangles.cu")
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC SAH_KD_TREE_STATIC_DEFINE)
endif()
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        "libsah_kd_tree"
        "libscene_loader"
    PUBLIC
        Qt6::Core)

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
