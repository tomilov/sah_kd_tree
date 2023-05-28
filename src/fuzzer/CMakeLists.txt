#[[
to debug some test input run
with -fork=0 command line option
]]
project("fuzzer")

if(NOT ${SAH_KD_TREE_BUILD_FUZZER})
    return()
endif()

add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cu"
        "${PROJECT_NAME}.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "libsah_kd_tree"
        "libutils")
target_compile_options(
    "${PROJECT_NAME}"
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fsanitize=fuzzer>>)
target_link_options(
    "${PROJECT_NAME}"
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fsanitize=fuzzer>>)

add_custom_command(
    TARGET
        "${PROJECT_NAME}"
    POST_BUILD
    COMMENT
        "Create symlink to fuzzer"
    COMMAND
        "${CMAKE_COMMAND}" -E create_symlink "$<TARGET_FILE:${PROJECT_NAME}>" "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzzer")
