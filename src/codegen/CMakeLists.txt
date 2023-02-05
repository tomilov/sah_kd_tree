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

macro(add_codegen_library name dependency command_arg)
    add_custom_command(
        MAIN_DEPENDENCY
            "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
        DEPENDS
            "${dependency}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${name}.hpp.jinja2"
            "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp.jinja2"
        COMMAND
            #${CMAKE_COMMAND} -E env PYTHONPATH="${spirv_headers_path}"
            Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
                --source-dir="${CMAKE_CURRENT_SOURCE_DIR}"
                --clang-format-executable="${clang_format_executable}"
                --clang-format="${CMAKE_SOURCE_DIR}/.clang-format"
                "${name}"
                "${command_arg}"
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
    generate_export_header("lib${target_name}" BASE_NAME "${target_name}")
    if(NOT BUILD_SHARED_LIBS)
        string(TOUPPER "${target_name}" STATIC_DEFINE_PREFIX)
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
endmacro()

get_target_property(spirv_headers_path "SPIRV-Headers" INTERFACE_INCLUDE_DIRECTORIES)
add_codegen_library(
    "spirv_format"
    "${spirv_headers_path}/spirv/unified1/spirv.py"
    --spirv="${spirv_headers_path}")

target_link_libraries(
    "lib${PROJECT_NAME}_spirv_format"
    PRIVATE
        "SPIRV-Headers"
    PUBLIC
        "SPIRV-Reflect")

get_target_property(vulkan_registry_path Vulkan::Registry INTERFACE_INCLUDE_DIRECTORIES)
add_codegen_library(
    "vulkan_utils"
    "${vulkan_registry_path}/vk.xml"
    --vulkan-registry="${vulkan_registry_path}/vk.xml")

target_link_libraries(
    "lib${PROJECT_NAME}_vulkan_utils"
    PUBLIC
        "Vulkan-Headers")

