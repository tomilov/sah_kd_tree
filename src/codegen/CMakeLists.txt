project("codegen")

set(Python3_FIND_VIRTUALENV FIRST)
find_package(
    Python3
    REQUIRED
    COMPONENTS
        Interpreter)

find_program(
    clang_format_executable
    NAMES
        clang-format
    REQUIRED)

function(add_codegen_library name)
    cmake_parse_arguments(PARSE_ARGV 1 add_codegen_library "" "DEPENDENCY;COMMAND_ARG" "")
    add_custom_command(
        MAIN_DEPENDENCY
            "generator.py"
        DEPENDS
            "${add_codegen_library_DEPENDENCY}"
            "${CMAKE_SOURCE_DIR}/_clang-format"
            "${name}.hpp.jinja2"
            "${name}.cpp.jinja2"
        COMMAND
            #${CMAKE_COMMAND} -E env PYTHONPATH="${spirv_headers_path}"
            Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
                --source-dir="${CMAKE_CURRENT_SOURCE_DIR}"
                --clang-format-executable="${clang_format_executable}"
                --clang-format-config="${CMAKE_SOURCE_DIR}/_clang-format"
                "${name}"
                "${add_codegen_library_COMMAND_ARG}"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
                "${name}.hpp.tmp"
                "${name}.hpp"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
                "${name}.cpp.tmp"
                "${name}.cpp"
        COMMAND
            "${CMAKE_COMMAND}" -E rm
                "${name}.hpp.tmp"
                "${name}.cpp.tmp"
        OUTPUT
            "${name}.hpp"
            "${name}.cpp")

    set(target_name "${PROJECT_NAME}_${name}")
    add_library("lib${target_name}")
    set_target_properties(
        "lib${target_name}"
        PROPERTIES
            LIBRARY_OUTPUT_NAME "${target_name}"
            ARCHIVE_OUTPUT_NAME "${target_name}")
    generate_export_header("lib${target_name}" BASE_NAME "${name}")
    if(NOT BUILD_SHARED_LIBS)
        string(TOUPPER "${name}" STATIC_DEFINE_PREFIX)
        target_compile_definitions(
            "lib${target_name}"
            PUBLIC
                ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
    endif()
    target_sources(
        "lib${target_name}"
        PRIVATE
            "generator.py"
            "${name}.hpp.jinja2"
            "${name}.cpp.jinja2"
            "${name}.hpp"
            "${name}.cpp")
    target_link_libraries(
        "lib${target_name}"
        PRIVATE
            "libformat"
            "libutils")
endfunction()

set(spirv_headers_path "${SPIRV-Headers_SOURCE_DIR}/include")
add_codegen_library(
    "spirv_format"
    DEPENDENCY "${spirv_headers_path}/spirv/unified1/spirv.py"
    COMMAND_ARG --spirv-headers="${spirv_headers_path}")

target_link_libraries(
    "lib${PROJECT_NAME}_spirv_format"
    PRIVATE
        "SPIRV-Headers"
    PUBLIC
        "SPIRV-Reflect")
target_include_directories(
    "lib${PROJECT_NAME}_spirv_format"
    SYSTEM PUBLIC
        "${spirv_headers_path}")

get_target_property(vulkan_registry_path Vulkan::Headers VULKAN_REGISTRY_PATH)
add_codegen_library(
    "vulkan_utils"
    DEPENDENCY "${vulkan_registry_path}/vk.xml"
    COMMAND_ARG --vulkan-registry="${vulkan_registry_path}/vk.xml")

target_link_libraries(
    "lib${PROJECT_NAME}_vulkan_utils"
    PRIVATE
        "SPIRV-Reflect"
    PUBLIC
        "Vulkan-Headers")
target_include_directories(
    "lib${PROJECT_NAME}_vulkan_utils"
    SYSTEM PUBLIC
        "${spirv_headers_path}")

