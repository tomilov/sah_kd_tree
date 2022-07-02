#[[
to debug some test input run
with -fork=0 command line option
]]
project("fuzzer")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
    return()
endif()

option(BUILD_FUZZ "Whether to build 'fuzzer' target" ON)
if(NOT BUILD_FUZZ)
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
        "sah_kd_tree")
target_compile_options(
    "${PROJECT_NAME}"
    PRIVATE
        "-fsanitize=fuzzer")
target_link_libraries(
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
