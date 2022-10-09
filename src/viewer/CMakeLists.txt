project("viewer")

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

qt6_add_library("lib${PROJECT_NAME}" SHARED)
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
if(NOT BUILD_SHARED_LIBS)
    string(TOUPPER ${PROJECT_NAME} STATIC_DEFINE_PREFIX)
    target_compile_definitions(
        "lib${PROJECT_NAME}"
        PUBLIC
            ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
endif()
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "example_renderer.hpp"
        "example_renderer.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        "librenderer"
        Qt6::Gui
        Qt6::Qml
        Qt6::Quick)

file(
    GLOB_RECURSE
    shader_files
    LIST_DIRECTORIES FALSE
    "*.glsl")
set_source_files_properties(${shader_files} PROPERTIES HEADER_FILE_ONLY TRUE)

foreach(shader_stage IN ITEMS "frag" "vert" "tesc" "tese" "geom" "comp" "mesh" "task" "rgen" "rint" "rahit" "rchit" "rmiss" "rcall")
    list(APPEND stage_shader_globs "*.${shader_stage}.glsl")
endforeach()

file(
    GLOB_RECURSE
    stage_shader_files
    LIST_DIRECTORIES FALSE
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    ${stage_shader_globs})

foreach(stage_shader_file IN LISTS stage_shader_files)
    compile_shader_to_spirv("${stage_shader_file}")
endforeach()

source_group("Shader Files" REGULAR_EXPRESSION "\.glsl$")

target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        ${shader_files}
    )

qt6_add_qml_module(
    "lib${PROJECT_NAME}"
    URI SahKdTree
    VERSION 1.0
    RESOURCES
        "fullscreen_triangle.vert.spv"
        "fullscreen_triangle.frag.spv"
    RESOURCE_PREFIX "/qml/imports")


qt6_add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "main.cpp"
    )
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "lib${PROJECT_NAME}"
        Qt6::Gui
        Qt6::Qml
        Qt6::Widgets
        Qt6::Quick
        Qt6::QuickControls2)

qt6_add_resources(
    "${PROJECT_NAME}"
    "ui"
    PREFIX "/qml"
    FILES
        "ui.qml")

set(QML_IMPORT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/src/viewer")

source_group("UI Files" REGULAR_EXPRESSION "ui.qml$")

add_subdirectory("tests")
