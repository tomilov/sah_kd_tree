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
function(target_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 target_shaders "" "OUTPUT_VARIABLE" "SHADERS")
    foreach(shader_file IN LISTS target_shaders_SHADERS)
        target_sources(
            "${target}"
            PRIVATE
                "${shader_file}")

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
                output_file)
        add_custom_command(
            COMMENT
                "Build shader file ${shader_file} for stage ${stage}"
            MAIN_DEPENDENCY
                "${shader_file}"
            VERBATIM
            WORKING_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPFILE  # sadly not works
                "${output_file}.d"
            COMMAND
                Vulkan::glslangValidator
                ARGS
                    -gVS $<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,-g,-g0>
                    --target-env vulkan1.3
                    --spirv-val
                    "${shader_file}"
                    --depfile "${output_file}.d"
                    -o "${output_file}"
            OUTPUT
                "${CMAKE_CURRENT_SOURCE_DIR}/${output_file}") # full path is required because on Qt's side logic tied to full path
        target_sources(
            "${target}"
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/${output_file}")
        if(DEFINED target_shaders_OUTPUT_VARIABLE)
            list(APPEND "${target_shaders_OUTPUT_VARIABLE}" "${output_file}")
        endif()
    endforeach()
    if(DEFINED target_shaders_OUTPUT_VARIABLE)
        set("${target_shaders_OUTPUT_VARIABLE}" "${${target_shaders_OUTPUT_VARIABLE}}" PARENT_SCOPE)
    endif()
endfunction()
