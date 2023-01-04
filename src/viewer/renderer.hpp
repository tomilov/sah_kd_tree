#pragma once

#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QSizeF>
#include <QtGui/QVulkanDeviceFunctions>
#include <QtGui/QVulkanFunctions>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickWindow>

namespace viewer
{
class Renderer
{
public:
    Renderer(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr, vk::Device device, uint32_t queueFamilyIndex, vk::Queue queue);
    ~Renderer();

    void setT(float t);

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, QSizeF size);

private:
    enum class Stage
    {
        Vertex,
        Fragment,
    };

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
    vk::Device device;
    uint32_t queueFamilyIndex = 0;
    vk::Queue queue;

    QVulkanFunctions * instanceFunctions = nullptr;
    QVulkanDeviceFunctions * deviceFunctions = nullptr;

    float t = 0;

    QByteArray vertexShader;
    QByteArray fragmentShader;

    bool pipelineLayoutsAndDescriptorsInitialized = false;
    bool pipelinesInitialized = false;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    VkDeviceSize uniformBufferPerFrameSize = 0;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet uniformBufferDescriptorSet = VK_NULL_HANDLE;

    void prepareShader(Stage stage);
    void initPipelineLayouts(int framesInFlight);
    void initDescriptors();
    void initGraphicsPipelines(vk::RenderPass renderPass);
};

}  // namespace viewer
