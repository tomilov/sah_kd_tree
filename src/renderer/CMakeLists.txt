project("renderer")

set(CMAKE_CXX_STANDARD 20)

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
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "vma.hpp"
        "vma.cpp"
        "exception.hpp"
        "exception.cpp"
        "debug_utils.hpp"
        "debug_utils.cpp"
        "utils.hpp"
        "utils.cpp"
        "memory.hpp"
        "memory.cpp"
        "format.hpp"
        "format.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        "libsah_kd_tree"
        "libcommon"
        "gli"
        "VulkanMemoryAllocator"
    PUBLIC
        Vulkan::Headers
        "libscene"
        "libutils")

add_executable("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "main.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        "lib${PROJECT_NAME}")
target_compile_definitions(
    "${PROJECT_NAME}"
    PRIVATE
        APPLICATION_NAME="${PROJECT_NAME}")

add_subdirectory("tests")
