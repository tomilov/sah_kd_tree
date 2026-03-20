find_package(
    Vulkan
    REQUIRED COMPONENTS
        glslangValidator
)

list(
    APPEND stage_shader_extensions
    "vert"
    "tesc"
    "tese"
    "geom"
    "frag"
    "comp"
    "rgen"
    "rahit"
    "rchit"
    "rmiss"
    "rint"
    "rcall"
    "mesh"
    "task"
)
list(JOIN stage_shader_extensions "|" stage_shader_regex)
set(stage_shader_regex "\.(${stage_shader_regex})\.glsl")

find_program(spirv-val NAMES spirv-val)

# macros in Qt6CoreMacros.cmake don't allow to use files generated in binary dir as sources
# because of wierd logic
function(skt_target_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 "arg" "" "OUTPUT_VARIABLE" "SHADERS")
    if(DEFINED arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${PROJECT_NAME}: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    foreach(shader_file IN LISTS arg_SHADERS)
        target_sources(
            "${target}"
            PRIVATE
                "${shader_file}"
        )
        if(NOT shader_file MATCHES "${stage_shader_regex}")
            message(STATUS "Shader ${shader_file} is not stage file. Will not be compiled.")
            continue()
        endif()
        string(REGEX MATCH "${stage_shader_regex}" _ "${shader_file}")
        set(stage "${CMAKE_MATCH_1}")

        cmake_path(
            REPLACE_EXTENSION
                shader_file
            LAST_ONLY
            ".spv"
            OUTPUT_VARIABLE
                output_file
        )
        #get_filename_component(shader_file_extension "${shader_file}" NAME_WLE)
        #get_filename_component(shader_file_extension "${shader_file_extension}" LAST_EXT)
        #string(SUBSTRING "${shader_file_extension}" 1 -1 shader_file_extension)
        add_custom_command(
            COMMENT
                "Build shader file ${shader_file} for stage ${stage}"
            MAIN_DEPENDENCY
                "${shader_file}"
            DEPENDS
                "${CMAKE_SOURCE_DIR}/tools/fix_depfile.py"
            VERBATIM
            WORKING_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPFILE
                "${CMAKE_CURRENT_SOURCE_DIR}/${output_file}.d"
            COMMAND
                Vulkan::glslangValidator
                ARGS
                    -g
                    -gVS
                    --target-env vulkan1.4
                    --spirv-val
                    "${shader_file}"
                    --depfile "${output_file}.d"
                    -o "${output_file}"
            COMMAND
                spirv-val
                ARGS
                    --target-env vulkan1.4
                    --scalar-block-layout
                    "${output_file}"
            COMMAND
                Python3::Interpreter
                ARGS
                    "${CMAKE_SOURCE_DIR}/tools/fix_depfile.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "${output_file}.d"
            OUTPUT
                "${CMAKE_CURRENT_SOURCE_DIR}/${output_file}" # full path is required because on Qt's side logic tied to full path
        )
        target_sources(
            "${target}"
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/${output_file}"
        )
        if(DEFINED arg_OUTPUT_VARIABLE)
            list(APPEND "${arg_OUTPUT_VARIABLE}" "${output_file}")
        endif()
    endforeach()
    if(DEFINED arg_OUTPUT_VARIABLE)
        set("${arg_OUTPUT_VARIABLE}" "${${arg_OUTPUT_VARIABLE}}" PARENT_SCOPE)
    endif()
endfunction()
