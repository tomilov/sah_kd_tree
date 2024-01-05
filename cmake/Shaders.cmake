find_package(
    Vulkan
    REQUIRED
    COMPONENTS
        glslangValidator)

list(
    APPEND
        stage_shader_extensions
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
    "task")
list(JOIN stage_shader_extensions "|" stage_shader_regex)
set(stage_shader_regex "\.(${stage_shader_regex})\.glsl")

# macros in Qt6CoreMacros.cmake don't allow to use files generated in binary dir as sources
# because of wierd logic
function(compile_stage_shaders compiled_shader_files_debug_list_name compiled_shader_files_release_list_name)
    list(SUBLIST ARGV 2 -1 stage_shader_files)
    foreach(stage_shader_file IN LISTS stage_shader_files)
        if(NOT stage_shader_file MATCHES "${stage_shader_regex}")
            message(FATAL_ERROR "${stage_shader_file} is not shader stage file")
        endif()

        cmake_path(
            REPLACE_EXTENSION
                stage_shader_file
            LAST_ONLY
            ".debug.spv"
            OUTPUT_VARIABLE
                debug_output_file)
        add_custom_command(
            MAIN_DEPENDENCY
                "${stage_shader_file}"
            VERBATIM
            WORKING_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND
                Vulkan::glslangValidator
                ARGS
                    -gVS -g
                    --target-env vulkan1.3
                    --spirv-val
                    "${stage_shader_file}"
                    -o "${debug_output_file}"
            OUTPUT
                "${CMAKE_CURRENT_SOURCE_DIR}/${debug_output_file}") # full path is required because on Qt's side logic tied to full path
        list(APPEND compiled_shader_files_debug "${debug_output_file}")

        cmake_path(
            REPLACE_EXTENSION
                stage_shader_file
            LAST_ONLY
            ".spv"
            OUTPUT_VARIABLE
                release_output_file)
        add_custom_command(
            MAIN_DEPENDENCY
                "${stage_shader_file}"
            VERBATIM
            WORKING_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND
                Vulkan::glslangValidator
                ARGS
                    -gVS -g0
                    --target-env vulkan1.3
                    --spirv-val
                    "${stage_shader_file}"
                    -o "${release_output_file}"
            OUTPUT
                "${CMAKE_CURRENT_SOURCE_DIR}/${release_output_file}") # full path is required because on Qt's side logic tied to full path
        list(APPEND compiled_shader_files_release "${release_output_file}")
    endforeach()

    set("${compiled_shader_files_debug_list_name}" "${compiled_shader_files_debug}" PARENT_SCOPE)
    set("${compiled_shader_files_release_list_name}" "${compiled_shader_files_release}" PARENT_SCOPE)
endfunction()
