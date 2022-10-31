add_library("VulkanHeaders" INTERFACE)
target_include_directories(
    "VulkanHeaders"
    SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include")


add_library("VulkanInterface" INTERFACE)
target_compile_definitions(
    "VulkanInterface"
    INTERFACE
        VK_ENABLE_BETA_EXTENSIONS
        VULKAN_HPP_NO_DEFAULT_DISPATCHER
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VK_USE_PLATFORM_XCB_KHR)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(
        "VulkanInterface"
        INTERFACE
            VK_NO_PROTOTYPES)
else()
    target_link_libraries(
        "VulkanInterface"
        INTERFACE
            "vulkan")
endif()
target_link_libraries(
    "VulkanInterface"
    INTERFACE
        "VulkanHeaders")


add_library("VulkanMemoryAllocator" INTERFACE)
target_include_directories(
    "VulkanMemoryAllocator"
    SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator/include")
target_link_libraries(
    "VulkanMemoryAllocator"
    INTERFACE
        "VulkanInterface")


add_library("glmInterface" INTERFACE)
target_include_directories(
    "glmInterface"
    SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/gli/external/glm/..")
target_compile_definitions(
    "glmInterface"
    INTERFACE
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_EXPLICIT_CTOR
        GLM_FORCE_PURE)


add_library("gliInterface" INTERFACE)
target_include_directories(
    "gliInterface"
    SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/gli")
target_link_libraries(
    "gliInterface"
    INTERFACE
        "glmInterface")
