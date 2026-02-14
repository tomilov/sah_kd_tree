project("codegen")

find_program(
    clang_format_executable
    NAMES
        clang-format
    REQUIRED)

function(skt_add_codegen_library name)
    cmake_parse_arguments(PARSE_ARGV 1 "ARG" "" "DEPENDENCY;COMMAND_ARGUMENT;PRIVATE_LINKS;PUBLIC_LINKS;SYSTEM_PUBLIC_INCLUDES" "")
    add_custom_command(
        MAIN_DEPENDENCY
            "generator.py"
        DEPENDS
            "${ARG_DEPENDENCY}"
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
                "${ARG_COMMAND_ARGUMENT}"
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

    skt_add_common_library(
        "${PROJECT_NAME}_${name}"
        BASE_NAME
            "${name}"
        SOURCES
            "generator.py"
            "${name}.hpp.jinja2"
            "${name}.cpp.jinja2"
            "${name}.hpp"
            "${name}.cpp"
        PRIVATE_LINKS
            "libformat"
            "libutils"
            ${ARG_PRIVATE_LINKS}
        PUBLIC_LINKS
            ${ARG_PUBLIC_LINKS}
        SYSTEM_PUBLIC_INCLUDES
            ${ARG_SYSTEM_PUBLIC_INCLUDES})
endfunction()

set(spirv_headers_path "${SPIRV-Headers_SOURCE_DIR}/include")
skt_add_codegen_library(
    "spirv_format"
    DEPENDENCY
        "${spirv_headers_path}/spirv/unified1/spirv.py"
    COMMAND_ARGUMENT
        --spirv-headers="${spirv_headers_path}"
    PRIVATE_LINKS
        "SPIRV-Headers"
    PUBLIC_LINKS
        "SPIRV-Reflect"
    SYSTEM_PUBLIC_INCLUDES
        "${spirv_headers_path}")

target_include_directories(
    "lib${PROJECT_NAME}_spirv_format"
    SYSTEM PUBLIC
        "${spirv_headers_path}")

get_target_property(vulkan_registry_path Vulkan::Headers VULKAN_REGISTRY_PATH)
skt_add_codegen_library(
    "vulkan_utils"
    DEPENDENCY
        "${vulkan_registry_path}/vk.xml"
    COMMAND_ARGUMENT
        --vulkan-registry="${vulkan_registry_path}/vk.xml"
    PRIVATE_LINKS
        "SPIRV-Reflect"
    PUBLIC_LINKS
        Vulkan::Headers
    SYSTEM_PUBLIC_INCLUDES
        "${spirv_headers_path}")


