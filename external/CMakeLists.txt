set("VulkanHeaders_DIR" "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers" CACHE PATH "Vulkan-Headers source path")
add_library("VulkanHeaders" INTERFACE)
target_include_directories(
    "VulkanHeaders"
    SYSTEM INTERFACE
        "${VulkanHeaders_DIR}/include")
