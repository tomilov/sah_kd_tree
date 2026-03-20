project("codegen")

option(SAH_KD_TREE_CODEGEN_FAIL_ON_FORMAT_MISMATCH "Fail on generated C++ sources format mismatch" ON)

function(skt_add_codegen_library name)
    cmake_parse_arguments(
        PARSE_ARGV 1
        "arg"
        ""
        "DEPENDENCY;COMMAND_ARGUMENT;PRIVATE_LINKS;PUBLIC_LINKS;SYSTEM_PUBLIC_INCLUDES"
        ""
    )
    if(DEFINED arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${PROJECT_NAME}: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    add_custom_command(
        MAIN_DEPENDENCY
            "generator.py"
        DEPENDS
            "${arg_DEPENDENCY}"
            "${CMAKE_SOURCE_DIR}/_clang-format"
            "${name}.hpp.jinja2"
            "${name}.cpp.jinja2"
        COMMAND
            #${CMAKE_COMMAND} -E env PYTHONPATH="${spirv_headers_path}"
            Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
                --source-dir="${CMAKE_CURRENT_SOURCE_DIR}"
                --clang-format-executable="${CMAKE_SOURCE_DIR}/venv/bin/clang-format"
                --clang-format-config="${CMAKE_SOURCE_DIR}/_clang-format"
                $<$<BOOL:${SAH_KD_TREE_CODEGEN_FAIL_ON_FORMAT_MISMATCH}>:--fail-on-format-mismatch>
                "${name}"
                "${arg_COMMAND_ARGUMENT}"
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
            "${name}.cpp"
    )
    skt_add_library(
        TARGET "lib${PROJECT_NAME}_${name}"
        BASE_NAME "${name}"
        SOURCES
            "generator.py"
            "${name}.hpp.jinja2"
            "${name}.cpp.jinja2"
            "${name}.hpp"
            "${name}.cpp"
        PRIVATE_LINKS
            "libformat"
            ${arg_PRIVATE_LINKS}
        PUBLIC_LINKS
            ${arg_PUBLIC_LINKS}
        SYSTEM_PUBLIC_INCLUDES
            ${arg_SYSTEM_PUBLIC_INCLUDES}
    )
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
        "${spirv_headers_path}"
)

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
        "${spirv_headers_path}"
)


