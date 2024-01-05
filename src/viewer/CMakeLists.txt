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

target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        Qt6::Gui
        Qt6::Qml
        Qt6::Quick
        Qt6::Widgets
        "libformat"
        "libengine"
        "libscene"
        "libscene_loader"
        "gli"
    PUBLIC
        "libutils")

qt6_add_qml_module(
    "lib${PROJECT_NAME}"
    URI
        SahKdTree
    VERSION
        1.0
    DEPENDENCIES
        QtQuick
        QtQuick.Controls
        QtQuick.Window
        QtQuick.Layouts
        QtQuick.Dialogs
        Qt.labs.settings
        Qt.labs.folderlistmodel
    RESOURCE_PREFIX
        "/${CMAKE_PROJECT_NAME}/imports"
    SOURCES
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "file_io.hpp"
        "file_io.cpp"
        "scene_manager.hpp"
        "scene_manager.cpp"
        "engine_wrapper.hpp"
        "engine_wrapper.cpp"
        "renderer.hpp"
        "renderer.cpp"
        "shaders/uniform_buffer.glsl")

compile_stage_shaders(
    compiled_shader_files_debug
    compiled_shader_files_release
    #"shaders/uniform_buffer.glsl"
    "shaders/barycentric_color.frag.glsl"
    "shaders/fullscreen_rect.vert.glsl"
    "shaders/identity.vert.glsl"
    "shaders/squircle.frag.glsl")

foreach(compiled_shader_file_debug compiled_shader_file_release IN ZIP_LISTS compiled_shader_files_debug compiled_shader_files_release)
    set(resource_alias "${compiled_shader_file_release}")
    #cmake_path(RELATIVE_PATH compiled_shader_file_release OUTPUT_VARIABLE resource_alias)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(compiled_shader_file "${compiled_shader_file_debug}")
    else()
        set(compiled_shader_file "${compiled_shader_file_release}")
    endif()
    set_source_files_properties(
        "${compiled_shader_file}"
        DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
        PROPERTIES
            QT_RESOURCE_ALIAS "${resource_alias}")
    qt6_target_qml_sources(
        "lib${PROJECT_NAME}"
        RESOURCES
            "${compiled_shader_file}")
endforeach()

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
        "libformat"
        "libengine"
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
    PREFIX
        "/${CMAKE_PROJECT_NAME}"
    FILES
        "qml/qmldir"
        "qml/components/CenteredDialog.qml"
        "qml/ui.qml")

source_group(
    "UI Files"
    REGULAR_EXPRESSION
        "_?ui.qml$")

add_subdirectory("tests")
