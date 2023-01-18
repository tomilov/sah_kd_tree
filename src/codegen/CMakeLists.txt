project("codegen")

get_target_property(vulkan_registry_path Vulkan::Registry INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(spirv_headers_path "SPIRV-Headers" INTERFACE_INCLUDE_DIRECTORIES)

set(Python3_FIND_VIRTUALENV FIRST)
find_package(
    Python3
    REQUIRED
    COMPONENTS
        Interpreter)

find_program(clang_format_executable NAMES clang-format)

add_custom_command(
    MAIN_DEPENDENCY
        "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
    DEPENDS
        "${vulkan_registry_path}/vk.xml"
        "${spirv_headers_path}/spirv/unified1/spirv.py"
    COMMAND
        #${CMAKE_COMMAND} -E env PYTHONPATH="${spirv_headers_path}"
        Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
            --vulkan-registry="${vulkan_registry_path}/vk.xml"
            --spirv="${spirv_headers_path}"
            --output-dir="${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND
        "${clang_format_executable}"
            -i
            -style="file:${CMAKE_SOURCE_DIR}/.clang-format"
            "spirv_format_unformatted.hpp"
            "spirv_format_unformatted.cpp"
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different "spirv_format_unformatted.hpp" "spirv_format.hpp"
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different "spirv_format_unformatted.cpp" "spirv_format.cpp"
    COMMAND
        "${CMAKE_COMMAND}" -E rm "spirv_format_unformatted.hpp" "spirv_format_unformatted.cpp"
    OUTPUT
        "spirv_format.hpp"
        "spirv_format.cpp")

add_library("lib${PROJECT_NAME}")
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
        "generator.py"
        "spirv_format.hpp"
        "spirv_format.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PUBLIC
        "SPIRV-Reflect"
        "SPIRV-Headers"
        "Vulkan-Headers")
