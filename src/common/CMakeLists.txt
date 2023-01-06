project("common")

add_library("lib${PROJECT_NAME}")
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
if(NOT BUILD_SHARED_LIBS)
    string(TOUPPER ${PROJECT_NAME} STATIC_DEFINE_PREFIX)
    target_compile_definitions(
        "lib${PROJECT_NAME}"
        PUBLIC
            ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
endif()
target_sources(
    "lib${PROJECT_NAME}"
    PUBLIC
        "config.hpp")

find_package(Git REQUIRED)
execute_process(
    WORKING_DIRECTORY
        "${CMAKE_SOURCE_DIR}"
    COMMAND
        "${GIT_EXECUTABLE}" describe
            --always
            --tags
            --dirty
            --abbrev=40
    OUTPUT_VARIABLE
        PROJECT_GIT_HASH
    RESULTS_VARIABLE
        GIT_RETURN_CODE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT GIT_RETURN_CODE STREQUAL "0")
    message(FATAL_ERROR "Not a git repository")
endif()
env_or_default(PROJECT_GIT_HASH "${PROJECT_GIT_HASH}")
configure_file("version.hpp.in" "version.hpp")

target_sources(
    "lib${PROJECT_NAME}"
    PUBLIC
        "version.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version.hpp")
