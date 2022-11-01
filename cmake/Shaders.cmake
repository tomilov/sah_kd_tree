find_package(
    Vulkan
    REQUIRED
    COMPONENTS
        glslangValidator)

function(compile_shaders_to_spirv)
    file(
        GLOB_RECURSE
        shader_files
        LIST_DIRECTORIES FALSE
        "*.glsl")
    set_source_files_properties(
        ${shader_files}
        PROPERTIES
            HEADER_FILE_ONLY TRUE)

    list(
        APPEND stage_shader_globs
            "frag"
            "vert"
            "tesc"
            "tese"
            "geom"
            "comp"
            "mesh"
            "task"
            "rgen"
            "rint"
            "rahit"
            "rchit"
            "rmiss"
            "rcall")
    list(
        TRANSFORM stage_shader_globs
        REPLACE ".+" "*.\\0.glsl")

    file(
        GLOB_RECURSE
        stage_shader_files
        LIST_DIRECTORIES FALSE
        ${stage_shader_globs})

    find_program(SPIRV_OPT_EXECUTABLE "spirv-opt")
    find_program(SPIRV_VAL_EXECUTABLE "spirv-val")
    foreach(stage_shader_file IN LISTS stage_shader_files)
        cmake_path(
            REPLACE_EXTENSION stage_shader_file
            LAST_ONLY
            ".debug.spv"
            OUTPUT_VARIABLE debug_output_file)
        add_custom_command(
            MAIN_DEPENDENCY "${stage_shader_file}"
            VERBATIM
            COMMAND Vulkan::glslangValidator ARGS -g --target-env vulkan1.3 "${stage_shader_file}" -o "${debug_output_file}"
            COMMAND "${SPIRV_VAL_EXECUTABLE}" ARGS --scalar-block-layout --target-env vulkan1.3 "${debug_output_file}"
            OUTPUT "${debug_output_file}")

        cmake_path(
            REPLACE_EXTENSION stage_shader_file
            LAST_ONLY
            ".spv"
            OUTPUT_VARIABLE release_output_file)
        add_custom_command(
            MAIN_DEPENDENCY "${debug_output_file}"
            VERBATIM
            COMMAND "${SPIRV_OPT_EXECUTABLE}" ARGS --skip-validation -O --strip-debug "${debug_output_file}" -o "${release_output_file}"
            COMMAND "${SPIRV_VAL_EXECUTABLE}" ARGS --scalar-block-layout --target-env vulkan1.3 "${release_output_file}"
            OUTPUT "${release_output_file}")
    endforeach()
endfunction()