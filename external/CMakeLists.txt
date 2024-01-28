add_subdirectory("Vulkan-Headers" EXCLUDE_FROM_ALL SYSTEM)
set_target_properties(
    "Vulkan-Headers"
    PROPERTIES
        VULKAN_REGISTRY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/registry")
target_compile_definitions(
    Vulkan-Headers
    INTERFACE
        VK_ENABLE_BETA_EXTENSIONS
        VK_USE_PLATFORM_XCB_KHR
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_NO_DEFAULT_DISPATCHER
        VULKAN_HPP_STORAGE_SHARED
        VULKAN_HPP_NO_CONSTRUCTORS
        VULKAN_HPP_TYPESAFE_CONVERSION
        VULKAN_HPP_ASSERT=ASSERT)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(
        Vulkan-Headers
        INTERFACE
            VK_NO_PROTOTYPES)
else()
    target_link_libraries(
        Vulkan-Headers
        INTERFACE
            "vulkan")
endif()


add_library("VulkanMemoryAllocator" INTERFACE)
target_include_directories(
    "VulkanMemoryAllocator"
    SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator/include")
target_link_libraries(
    "VulkanMemoryAllocator"
    INTERFACE
        Vulkan::Headers)


add_subdirectory("gli" EXCLUDE_FROM_ALL SYSTEM)
target_compile_definitions(
    "gli"
    INTERFACE
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_EXPLICIT_CTOR
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_LEFT_HANDED)


set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_SYSTEM_HEADERS ON CACHE BOOL "" FORCE)
add_subdirectory("fmt" EXCLUDE_FROM_ALL SYSTEM)


set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_SYSTEM_INCLUDES ON CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED BUILD_SHARED_LIBS CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory("spdlog" EXCLUDE_FROM_ALL SYSTEM)


set(SPIRV_HEADERS_SKIP_EXAMPLES YES CACHE BOOL "" FORCE)
set(SPIRV_HEADERS_SKIP_INSTALL YES CACHE BOOL "" FORCE)
add_subdirectory("SPIRV-Headers" EXCLUDE_FROM_ALL SYSTEM)


add_library("SPIRV-Reflect" STATIC)
target_sources(
    "SPIRV-Reflect"
    PRIVATE
        "SPIRV-Reflect/spirv_reflect.h"
        "SPIRV-Reflect/spirv_reflect.cpp"
        "SPIRV-Reflect/common/output_stream.h"
        "SPIRV-Reflect/common/output_stream.cpp")
target_include_directories(
    "SPIRV-Reflect"
    SYSTEM PUBLIC
        "SPIRV-Reflect")


add_subdirectory("nlohmann")


set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
set(TINYEXR_USE_MINIZ ON CACHE BOOL "" FORCE)
add_subdirectory("tinyexr" EXCLUDE_FROM_ALL SYSTEM)


set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("tinygltf" EXCLUDE_FROM_ALL SYSTEM)
