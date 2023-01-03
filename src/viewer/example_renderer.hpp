#pragma once

#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QSizeF>
#include <QtGui/QVulkanDeviceFunctions>
#include <QtGui/QVulkanFunctions>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickWindow>

#include <functional>

namespace viewer
{
class ExampleRenderer
{
public:
    using GetInstanceProcAddress = std::function<PFN_vkVoidFunction(const char * name)>;

    ExampleRenderer(GetInstanceProcAddress getInstanceProcAddress, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, vk::Queue queue);
    ~ExampleRenderer();

    void setT(qreal t)
    {
        m_t = t;
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, QSizeF size);

private:
    enum Stage
    {
        VertexStage,
        FragmentStage
    };

    GetInstanceProcAddress getInstanceProcAddress;
    QVulkanInstance * instance = nullptr;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex = 0;
    vk::Queue queue;

    qreal m_t = 0;

    QByteArray m_vert;
    QByteArray m_frag;

    bool m_initialized = false;
    bool pipelineInitialized = false;
    QVulkanDeviceFunctions * m_devFuncs = nullptr;
    QVulkanFunctions * m_funcs = nullptr;

    VkBuffer m_vbuf = VK_NULL_HANDLE;
    VkDeviceMemory m_vbufMem = VK_NULL_HANDLE;
    VkBuffer m_ubuf = VK_NULL_HANDLE;
    VkDeviceMemory m_ubufMem = VK_NULL_HANDLE;
    VkDeviceSize m_allocPerUbuf = 0;

    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_resLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ubufDescriptor = VK_NULL_HANDLE;

    void prepareShader(Stage stage);
    void initPipelineLayouts(int framesInFlight);
    void initPipelines(vk::RenderPass renderPass);
};

}  // namespace viewer
