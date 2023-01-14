project("engine")

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
        "fwd.hpp"
        "vma.hpp"
        "vma.cpp"
        "exception.hpp"
        "exception.cpp"
        "debug_utils.hpp"
        "debug_utils.cpp"
        "utils.hpp"
        "utils.cpp"
        "types.hpp"
        "types.cpp"
        "file_io.hpp"
        "file_io.cpp"
        "memory.hpp"
        "memory.cpp"
        "format.hpp"
        "format.cpp"
        "library.hpp"
        "library.cpp"
        "instance.hpp"
        "instance.cpp"
        "physical_device.hpp"
        "physical_device.cpp"
        "fence.hpp"
        "fence.cpp"
        "device.hpp"
        "device.cpp"
        "command_buffer.hpp"
        "command_buffer.cpp"
        "command_pool.hpp"
        "command_pool.cpp"
        "queue.hpp"
        "queue.cpp"
        "shader_module.hpp"
        "shader_module.cpp"
        "render_pass.hpp"
        "render_pass.cpp"
        "framebuffer.hpp"
        "framebuffer.cpp"
        "pipeline_cache.hpp"
        "pipeline_cache.cpp"
        "graphics_pipeline.hpp"
        "graphics_pipeline.cpp"
        "spirv_reflect_dump.hpp"
        "spirv_reflect_dump.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        "libsah_kd_tree"
        "libcommon"
        "gli"
        "VulkanMemoryAllocator"
        "SPIRV-Reflect"
        "libcodegen"
        "VkLayer_utils" # from system
        "nlohmann_json"
    PUBLIC
        Vulkan::Headers
        "libutils")

set_source_files_properties(
    "spirv_reflect_dump.cpp"
    PROPERTIES
        COMPILE_FLAGS
            "-Werror=switch")

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

if(${ipo_is_supported})
    set_target_properties(
        "lib${PROJECT_NAME}"
        "${PROJECT_NAME}"
        PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
