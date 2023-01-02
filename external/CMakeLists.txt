add_subdirectory("Vulkan-Headers" EXCLUDE_FROM_ALL SYSTEM)
target_compile_definitions(
    Vulkan-Headers
    INTERFACE
        VK_ENABLE_BETA_EXTENSIONS
        VK_USE_PLATFORM_XCB_KHR
        VULKAN_HPP_NO_DEFAULT_DISPATCHER
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)
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
        GLM_FORCE_PURE)


add_subdirectory("fmt" EXCLUDE_FROM_ALL SYSTEM)


set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_SYSTEM_INCLUDES ON)
add_subdirectory("spdlog" EXCLUDE_FROM_ALL SYSTEM)


add_library("SPIRV-Reflect" STATIC)
target_sources(
    "SPIRV-Reflect"
    PRIVATE
        "SPIRV-Reflect/spirv_reflect.h"
        "SPIRV-Reflect/spirv_reflect.cpp")
target_include_directories(
    "SPIRV-Reflect"
    PUBLIC
        "SPIRV-Reflect")
