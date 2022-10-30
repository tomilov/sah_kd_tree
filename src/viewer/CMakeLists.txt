project("viewer")

qt6_add_library("lib${PROJECT_NAME}" SHARED) # cannot be STATIC
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
        "libutils"
        Qt6::Gui
        Qt6::Qml
        Qt6::Quick)

compile_shaders_to_spirv()
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Default")
    qt6_add_qml_module(
        "lib${PROJECT_NAME}"
        URI SahKdTree
        VERSION 1.0
        SOURCES
            "fullscreen_triangle.vert.glsl"
            "fullscreen_triangle.frag.glsl"
        RESOURCES
            "fullscreen_triangle.vert.debug.spv"
            "fullscreen_triangle.frag.debug.spv"
        RESOURCE_PREFIX "/qml/imports")
else()
    qt6_add_qml_module(
        "lib${PROJECT_NAME}"
        URI SahKdTree
        VERSION 1.0
        SOURCES
            "fullscreen_triangle.vert.glsl"
            "fullscreen_triangle.frag.glsl"
        RESOURCES
            "fullscreen_triangle.vert.spv"
            "fullscreen_triangle.frag.spv"
        RESOURCE_PREFIX "/qml/imports")
endif()

qt6_add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "main.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "lib${PROJECT_NAME}"
        "libutils"
        "libcommon"
        Qt6::Gui
        Qt6::Qml
        Qt6::Widgets
        Qt6::Quick
        Qt6::QuickControls2)
target_compile_definitions(
    "${PROJECT_NAME}"
    PRIVATE
        APPLICATION_NAME="${PROJECT_NAME}")

qt6_add_resources(
    "${PROJECT_NAME}"
    "ui"
    PREFIX "/qml"
    FILES
        "ui.qml")

source_group("UI Files" REGULAR_EXPRESSION "_?ui.qml$")

add_subdirectory("tests")
