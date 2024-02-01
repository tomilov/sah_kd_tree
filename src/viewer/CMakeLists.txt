project("viewer")

qt6_add_qml_module(
    "${PROJECT_NAME}_item"
    URI
        "SahKdTree"
    VERSION
        1.0
    RESOURCE_PREFIX
        "/${CMAKE_PROJECT_NAME}/imports"
    OUTPUT_DIRECTORY
        "SahKdTree")
#set_target_properties("${PROJECT_NAME}_item" PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/SahKdTree")
set(QML_IMPORT_PATH "${CMAKE_CURRENT_BINARY_DIR}/SahKdTree" CACHE STRING "" FORCE)

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
        "file_io.hpp"
        "file_io.cpp"
        "scene_manager.hpp"
        "scene_manager.cpp"
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

target_shaders(
    "${PROJECT_NAME}_item"
    SHADERS
        "shaders/uniform_buffer.glsl"
        "shaders/barycentric_color.frag.glsl"
        "shaders/fullscreen_rect.vert.glsl"
        "shaders/identity.vert.glsl"
        "shaders/squircle.frag.glsl"
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
