#[[
to debug some test input run
with -fork=0 command line option
]]
project("fuzzer")

if(NOT BUILD_FUZZER)
    return()
endif()

add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "libsah_kd_tree")
target_compile_options(
    "${PROJECT_NAME}"
    PRIVATE
        "-fsanitize=fuzzer")
target_link_options(
    "${PROJECT_NAME}"
    PRIVATE
        "-fsanitize=fuzzer")

if(UNIX)
    add_custom_command(
        TARGET
            "${PROJECT_NAME}"
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E create_symlink "$<TARGET_FILE:${PROJECT_NAME}>" "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzzer"
        COMMENT
            "Created symlink to fuzzer")
endif()
