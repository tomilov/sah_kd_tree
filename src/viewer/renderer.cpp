#include <common/version.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <viewer/renderer.hpp>
#include <viewer/resource_manager.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QIODevice>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtCore/QString>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QSGRendererInterface>

#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerRendererCategory)
Q_LOGGING_CATEGORY(viewerRendererCategory, "viewer.renderer")
}  // namespace

struct Renderer::Impl
{
    enum class Stage
    {
        Vertex,
        Fragment,
    };

    engine::Engine & engine;
    ResourceManager & resourceManager;

    engine::Library & library_NEW = *utils::CheckedPtr(engine.library.get());
    engine::Instance & instance_NEW = *utils::CheckedPtr(engine.instance.get());
    engine::PhysicalDevices & physicalDevices_NEW = *utils::CheckedPtr(engine.physicalDevices.get());
    engine::Device & device_NEW = *utils::CheckedPtr(engine.device.get());

    std::shared_ptr<const Resources> resources;
    std::unique_ptr<const Resources::GraphicsPipeline> graphicsPipeline_NEW;

    // TODO: descriptor allocator + descriptor layout cache

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
    vk::Device device;
    uint32_t queueFamilyIndex = 0;
    vk::Queue queue;

    QVulkanFunctions * instanceFunctions = nullptr;
    QVulkanDeviceFunctions * deviceFunctions = nullptr;

    UniformBuffer uniformBuffer_NEW;

    QByteArray vertexShader;
    QByteArray fragmentShader;

    bool pipelineLayoutsAndDescriptorsInitialized = false;
    bool pipelinesInitialized = false;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    vk::DeviceSize uniformBufferPerFrameSize = 0;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet uniformBufferDescriptorSet = VK_NULL_HANDLE;

    Impl(engine::Engine & engine, ResourceManager & resourceManager, PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr, vk::Device device,
         uint32_t queueFamilyIndex, vk::Queue queue)
        : engine{engine}
        , resourceManager{resourceManager}
        , vkGetInstanceProcAddr{vkGetInstanceProcAddr}
        , instance{instance->vkInstance()}
        , physicalDevice{physicalDevice}
        , vkGetDeviceProcAddr{vkGetDeviceProcAddr}
        , device{device}
        , queueFamilyIndex{queueFamilyIndex}
        , queue{queue}
    {
        Q_ASSERT(instance);
        Q_ASSERT(instance->isValid());
        Q_ASSERT(physicalDevice);
        Q_ASSERT(device);

        instanceFunctions = instance->functions();
        Q_ASSERT(instanceFunctions);
        deviceFunctions = instance->deviceFunctions(device);
        Q_ASSERT(deviceFunctions);

        init();
    }

    ~Impl()
    {
        if (!deviceFunctions) return;

        deviceFunctions->vkDestroyPipeline(device, graphicsPipeline, nullptr);
        deviceFunctions->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        deviceFunctions->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        deviceFunctions->vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        deviceFunctions->vkDestroyPipelineCache(device, pipelineCache, nullptr);

        deviceFunctions->vkDestroyBuffer(device, vertexBuffer, nullptr);
        deviceFunctions->vkFreeMemory(device, vertexBufferMemory, nullptr);

        deviceFunctions->vkDestroyBuffer(device, uniformBuffer, nullptr);
        deviceFunctions->vkFreeMemory(device, uniformBufferMemory, nullptr);
    }

    void setT(float t)
    {
        uniformBuffer_NEW.t = t;
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size);

private:
    void init();
    void prepareShader(Stage stage);
    void initPipelineLayouts(int framesInFlight);
    void initDescriptors();
    void initGraphicsPipelines(vk::RenderPass renderPass);
};

Renderer::Renderer(engine::Engine & engine, ResourceManager & resourceManager, PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr,
                   vk::Device device, uint32_t queueFamilyIndex, vk::Queue queue)
    : impl_{engine, resourceManager, vkGetInstanceProcAddr, instance, physicalDevice, vkGetDeviceProcAddr, device, queueFamilyIndex, queue}
{}

Renderer::~Renderer() = default;

void Renderer::setT(float t)
{
    return impl_->setT(t);
}
void Renderer::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    return impl_->frameStart(graphicsStateInfo);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size)
{
    return impl_->render(commandBuffer, renderPass, graphicsStateInfo, size);
}

void Renderer::Impl::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    if ((true)) {
        uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
        if (!resources || (resources->getFramesInFlight() != framesInFlight)) {
            graphicsPipeline_NEW = nullptr;
            resources = resourceManager.getOrCreateResources(framesInFlight);

            std::copy_n(std::data(kVertices), std::size(kVertices), resources->getUniformBuffer().map<VertexType>().get());
        }

        auto uniformBufferPerFrameSize = resources->getUniformBufferPerFrameSize();
        uint32_t uniformBufferIndex = utils::autoCast(graphicsStateInfo.currentFrameSlot);
        *resources->getUniformBuffer().map<UniformBuffer>(uniformBufferPerFrameSize * uniformBufferIndex, uniformBufferPerFrameSize).get() = uniformBuffer_NEW;
    } else {
        if (!pipelineLayoutsAndDescriptorsInitialized) {
            pipelineLayoutsAndDescriptorsInitialized = true;
            initPipelineLayouts(graphicsStateInfo.framesInFlight);
            initDescriptors();
        }
        VkDeviceSize uniformBufferOffset = graphicsStateInfo.currentFrameSlot * uniformBufferPerFrameSize;
        void * p = nullptr;
        VkResult err = deviceFunctions->vkMapMemory(device, uniformBufferMemory, uniformBufferOffset, uniformBufferPerFrameSize, 0, &p);
        if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to map uniform buffer memory: %d", err);
        INVARIANT(p, "Just successfully initialized");
        memcpy(p, &uniformBuffer_NEW, sizeof uniformBuffer_NEW);
        deviceFunctions->vkUnmapMemory(device, uniformBufferMemory);
    }
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size)
{
    if ((true)) {
        if (!resources) {
            return;
        }
        if (!graphicsPipeline_NEW || (graphicsPipeline_NEW->pipelineLayout.renderPass != renderPass)) {
            graphicsPipeline_NEW = resources->createGraphicsPipeline(renderPass);
        }

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_NEW->pipelines.pipelines.at(0), library_NEW.dispatcher);

        std::vector<vk::Buffer> vertexBuffers = {
            resources->getVertexBuffer().getBuffer(),
        };
        std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
        commandBuffer.bindVertexBuffers(0, vertexBuffers, vertexBufferOffsets, library_NEW.dispatcher);  // read about bindVertexBuffers2

        std::vector<uint32_t> dinamicOffsets = {
            uint32_t(utils::autoCast(uniformBufferPerFrameSize)) * uint32_t(utils::autoCast(graphicsStateInfo.currentFrameSlot)),
        };
        uint32_t firstSet = 0;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipeline_NEW->pipelineLayout.pipelineLayout, firstSet, resources->getDescriptorSets(), dinamicOffsets, library_NEW.dispatcher);

        std::vector<vk::Viewport> viewports = {
            {
                .x = 0,
                .y = 0,
                .width = utils::autoCast(size.width()),
                .height = utils::autoCast(size.height()),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            },
        };
        commandBuffer.setViewport(0, viewports, library_NEW.dispatcher);

        std::vector<vk::Rect2D> scissors = {
            {
                vk::Offset2D{.x = 0, .y = 0},
                vk::Extent2D{.width = utils::autoCast(size.width()), .height = utils::autoCast(size.height())},
            },
        };
        commandBuffer.setScissor(0, scissors, library_NEW.dispatcher);

        commandBuffer.draw(4, 1, 0, 0, library_NEW.dispatcher);
    } else {
        if (!pipelinesInitialized) {
            pipelinesInitialized = true;
            initGraphicsPipelines(renderPass);
        }

        VkCommandBuffer cb = commandBuffer;
        Q_ASSERT(cb);

        deviceFunctions->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkDeviceSize vbufOffset = 0;
        deviceFunctions->vkCmdBindVertexBuffers(cb, 0, 1, &vertexBuffer, &vbufOffset);

        uint32_t dynamicOffset = uniformBufferPerFrameSize * uint32_t(utils::autoCast(graphicsStateInfo.currentFrameSlot));
        deviceFunctions->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uniformBufferDescriptorSet, 1, &dynamicOffset);

        VkViewport vp = {0, 0, utils::autoCast(size.width()), utils::autoCast(size.height()), 0.0f, 1.0f};
        deviceFunctions->vkCmdSetViewport(cb, 0, 1, &vp);
        VkRect2D scissor = {{0, 0}, {utils::autoCast(size.width()), utils::autoCast(size.height())}};
        deviceFunctions->vkCmdSetScissor(cb, 0, 1, &scissor);

        deviceFunctions->vkCmdDraw(cb, 4, 1, 0, 0);
    }
}

void Renderer::Impl::init()
{
    if (vertexShader.isEmpty()) {
        prepareShader(Stage::Vertex);
        Q_ASSERT(!vertexShader.isEmpty());
    }
    if (fragmentShader.isEmpty()) {
        prepareShader(Stage::Fragment);
        Q_ASSERT(!fragmentShader.isEmpty());
    }
}

void Renderer::Impl::prepareShader(Stage stage)
{
    const auto kUri = u"SahKdTree"_s;
    QString stageName;
    switch (stage) {
    case Stage::Vertex: {
        stageName = u"vert"_s;
        break;
    }
    case Stage::Fragment: {
        stageName = u"frag"_s;
        break;
    }
    }
    INVARIANT(!stageName.isEmpty(), "Unknown stage {}", fmt::underlying(stage));
    QFile shaderFile{QLatin1String(":/%1/imports/%2/shaders/fullscreen_triangle.%3.spv").arg(QString::fromUtf8(sah_kd_tree::kProjectName), kUri, stageName)};
    if (!shaderFile.open(QIODevice::ReadOnly)) {
        QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to read shader %s", qPrintable(shaderFile.fileName()));
    }
    switch (stage) {
    case Stage::Vertex: {
        vertexShader = shaderFile.readAll();
        break;
    }
    case Stage::Fragment: {
        fragmentShader = shaderFile.readAll();
        break;
    }
    }
}

static inline VkDeviceSize aligned(VkDeviceSize v, VkDeviceSize byteAlign)
{
    return (v + byteAlign - 1) & ~(byteAlign - 1);
}

void Renderer::Impl::initPipelineLayouts(int framesInFlight)
{
    Q_ASSERT(framesInFlight <= 3);

    VkPhysicalDeviceProperties physDevProps = {};
    instanceFunctions->vkGetPhysicalDeviceProperties(physicalDevice, &physDevProps);

    VkPhysicalDeviceMemoryProperties physDevMemProps = {};
    instanceFunctions->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physDevMemProps);

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof kVertices;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VkResult err = deviceFunctions->vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create vertex buffer: %d", err);

    VkMemoryRequirements memReq = {};
    deviceFunctions->vkGetBufferMemoryRequirements(device, vertexBuffer, &memReq);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;

    uint32_t memTypeIndex = uint32_t(-1);
    const VkMemoryType * memType = physDevMemProps.memoryTypes;
    for (uint32_t i = 0; i < physDevMemProps.memoryTypeCount; ++i) {
        if (memReq.memoryTypeBits & (1 << i)) {
            if ((memType[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memType[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }
    if (memTypeIndex == uint32_t(-1)) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to find host visible and coherent memory type");

    allocInfo.memoryTypeIndex = memTypeIndex;
    err = deviceFunctions->vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to allocate vertex buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    void * p = nullptr;
    err = deviceFunctions->vkMapMemory(device, vertexBufferMemory, 0, allocInfo.allocationSize, 0, &p);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to map vertex buffer memory: %d", err);
    INVARIANT(p, "Just successfully initialized");
    memcpy(p, kVertices, sizeof kVertices);
    deviceFunctions->vkUnmapMemory(device, vertexBufferMemory);
    err = deviceFunctions->vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to bind vertex buffer memory: %d", err);

    uniformBufferPerFrameSize = aligned(sizeof uniformBuffer_NEW, physDevProps.limits.minUniformBufferOffsetAlignment);

    bufferInfo.size = framesInFlight * uniformBufferPerFrameSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    err = deviceFunctions->vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffer);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create uniform buffer: %d", err);
    deviceFunctions->vkGetBufferMemoryRequirements(device, uniformBuffer, &memReq);
    memTypeIndex = -1;
    for (uint32_t i = 0; i < physDevMemProps.memoryTypeCount; ++i) {
        if (memReq.memoryTypeBits & (1 << i)) {
            if ((memType[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memType[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }
    if (memTypeIndex == uint32_t(-1)) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to find host visible and coherent memory type");

    allocInfo.allocationSize = framesInFlight * uniformBufferPerFrameSize;
    allocInfo.memoryTypeIndex = memTypeIndex;
    err = deviceFunctions->vkAllocateMemory(device, &allocInfo, nullptr, &uniformBufferMemory);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to allocate uniform buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    err = deviceFunctions->vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to bind uniform buffer memory: %d", err);

    VkPipelineCacheCreateInfo pipelineCacheInfo = {};
    pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    err = deviceFunctions->vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create pipeline cache: %d", err);

    VkDescriptorSetLayoutBinding descLayoutBinding = {};
    descLayoutBinding.binding = 0;
    descLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    descLayoutBinding.descriptorCount = 1;
    descLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &descLayoutBinding;
    err = deviceFunctions->vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create descriptor set layout: %d", err);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    err = deviceFunctions->vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    if (err != VK_SUCCESS) qWarning("Failed to create pipeline layout: %d", err);
}

void Renderer::Impl::initDescriptors()
{
    VkDescriptorPoolSize descPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
    };
    VkDescriptorPoolCreateInfo descPoolInfo = {};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.flags = 0;  // won't use vkFreeDescriptorSets
    descPoolInfo.maxSets = 1;
    descPoolInfo.poolSizeCount = sizeof(descPoolSizes) / sizeof(descPoolSizes[0]);
    descPoolInfo.pPoolSizes = descPoolSizes;
    VkResult err = deviceFunctions->vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descriptorPool);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create descriptor pool: %d", err);

    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptorSetLayout;
    err = deviceFunctions->vkAllocateDescriptorSets(device, &descAllocInfo, &uniformBufferDescriptorSet);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to allocate descriptor set");

    VkWriteDescriptorSet writeInfo = {};
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstSet = uniformBufferDescriptorSet;
    writeInfo.dstBinding = 0;
    writeInfo.descriptorCount = 1;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = uniformBuffer;
    bufInfo.offset = 0;  // dynamic offset is used so this is ignored
    bufInfo.range = sizeof uniformBuffer_NEW;
    writeInfo.pBufferInfo = &bufInfo;
    deviceFunctions->vkUpdateDescriptorSets(device, 1, &writeInfo, 0, nullptr);
}

void Renderer::Impl::initGraphicsPipelines(vk::RenderPass renderPass)
{
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = std::size(vertexShader);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(vertexShader.constData());
    VkShaderModule vertShaderModule = {};
    VkResult err = deviceFunctions->vkCreateShaderModule(device, &shaderInfo, nullptr, &vertShaderModule);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create vertex shader module: %d", err);

    shaderInfo.codeSize = std::size(fragmentShader);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(fragmentShader.constData());
    VkShaderModule fragShaderModule = {};
    err = deviceFunctions->vkCreateShaderModule(device, &shaderInfo, nullptr, &fragShaderModule);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create fragment shader module: %d", err);

    VkPipelineShaderStageCreateInfo stageInfo[2] = {};
    stageInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stageInfo[0].module = vertShaderModule;
    stageInfo[0].pName = "main";
    stageInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stageInfo[1].module = fragShaderModule;
    stageInfo[1].pName = "main";
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stageInfo;

    VkVertexInputBindingDescription vertexBinding = {
        0,                  // binding
        2 * sizeof(float),  // stride
        VK_VERTEX_INPUT_RATE_VERTEX,
    };
    VkVertexInputAttributeDescription vertexAttr = {
        0,                        // location
        0,                        // binding
        VK_FORMAT_R32G32_SFLOAT,  // 'vertices' only has 2 floats per vertex
        0                         // offset
    };
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &vertexAttr;
    pipelineInfo.pVertexInputState = &vertexInputInfo;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicInfo = {};
    dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicInfo.dynamicStateCount = 2;
    dynamicInfo.pDynamicStates = dynStates;
    pipelineInfo.pDynamicState = &dynamicInfo;

    VkPipelineViewportStateCreateInfo viewportInfo = {};
    viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportInfo.viewportCount = viewportInfo.scissorCount = 1;
    pipelineInfo.pViewportState = &viewportInfo;

    VkPipelineInputAssemblyStateCreateInfo iaInfo = {};
    iaInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    pipelineInfo.pInputAssemblyState = &iaInfo;

    VkPipelineRasterizationStateCreateInfo rsInfo = {};
    rsInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsInfo.lineWidth = 1.0f;
    pipelineInfo.pRasterizationState = &rsInfo;

    VkPipelineMultisampleStateCreateInfo msInfo = {};
    msInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pipelineInfo.pMultisampleState = &msInfo;

    VkPipelineDepthStencilStateCreateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    pipelineInfo.pDepthStencilState = &dsInfo;

    VkPipelineColorBlendStateCreateInfo blendInfo = {};
    blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState blend = {};
    blend.blendEnable = true;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendInfo.attachmentCount = 1;
    blendInfo.pAttachments = &blend;
    pipelineInfo.pColorBlendState = &blendInfo;

    pipelineInfo.layout = pipelineLayout;

    pipelineInfo.renderPass = renderPass;

    err = deviceFunctions->vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &graphicsPipeline);

    deviceFunctions->vkDestroyShaderModule(device, vertShaderModule, nullptr);
    deviceFunctions->vkDestroyShaderModule(device, fragShaderModule, nullptr);

    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(viewerRendererCategory, QtFatalMsg).fatal("Failed to create graphics pipeline: %d", err);
}

}  // namespace viewer
