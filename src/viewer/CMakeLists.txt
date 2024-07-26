project("viewer")

set(APPEND QML_IMPORT_PATH "${CMAKE_CURRENT_BINARY_DIR}/SahKdTree" CACHE STRING "" FORCE)

qt6_add_qml_module(
    "${PROJECT_NAME}_item"
    URI
        "SahKdTree"
    VERSION
        1.0
    RESOURCE_PREFIX
        "/${CMAKE_PROJECT_NAME}/imports"
    OUTPUT_DIRECTORY
        "SahKdTree"
    IMPORT_PATH
        "SahKdTree"
    DEPENDENCIES
        QtCore
        QtQuick
        QtQuick.Controls
        QtQuick.Window
        QtQuick.Layouts
        QtQuick.Dialogs
        QtQuick3D)
set_target_properties("${PROJECT_NAME}_item" PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/SahKdTree")

generate_export_header("${PROJECT_NAME}_item" BASE_NAME "${PROJECT_NAME}")
if(NOT BUILD_SHARED_LIBS)
    string(TOUPPER ${PROJECT_NAME} STATIC_DEFINE_PREFIX)
    target_compile_definitions(
        "${PROJECT_NAME}_item"
        PUBLIC
            ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
endif()

target_sources(
    "${PROJECT_NAME}_item"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "render_node.hpp"
        "render_node.cpp"
        "file_io.hpp"
        "file_io.cpp"
        "scenes.hpp"
        "scenes.cpp"
        "engine.hpp"
        "engine.cpp"
        "descriptors.hpp"
        "descriptors.cpp"
        "pipelines.hpp"
        "pipelines.cpp"
        "engine_wrapper.hpp"
        "engine_wrapper.cpp"
        "renderer.hpp"
        "renderer.cpp"
        "utils.hpp"
        "utils.cpp")
target_link_libraries(
    "${PROJECT_NAME}_item"
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::GuiPrivate
        Qt6::Qml
        Qt6::Quick
        Qt6::Widgets
        Qt6::Svg
        Qt6::Xml
        Qt6::Quick3D
        "libformat"
        "libengine"
        "libcodegen_vulkan_utils"
        "libscene_data"
        "libscene_loader"
        "libdebug_utils"
        "libbuilder"
        "gli"
    PUBLIC
        "libutils")

target_shaders(
    "${PROJECT_NAME}_item"
    SHADERS
        "shaders/uniform_buffer.glsl"
        "shaders/barycentric_color.frag.glsl"
        "shaders/fullscreen_rect.vert.glsl"
        "shaders/identity.vert.glsl"
        "shaders/offscreen.frag.glsl"
    OUTPUT_VARIABLE
        compiled_shaders)

qt6_target_qml_sources(
    "${PROJECT_NAME}_item"
    RESOURCES
        ${compiled_shaders})

qt6_add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "application.hpp"
        "application.cpp"
        "main.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_item"
        "libutils"
        "libcommon"
        "libformat"
        "libengine"
        Qt6::Gui
        Qt6::GuiPrivate
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
        "qml/components/SceneOpenDialog.qml"
        "qml/ui.qml"
        "qml/utils.js")

source_group(
    "UI Files"
    REGULAR_EXPRESSION
        "$(.*_)?ui.qml$")

add_subdirectory("tests")
