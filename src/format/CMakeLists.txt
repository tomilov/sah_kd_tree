project("format")

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
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "vulkan.hpp"
        "vulkan.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PUBLIC
        Vulkan::Headers)

add_subdirectory("tests")
