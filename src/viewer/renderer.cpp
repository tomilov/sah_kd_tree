#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/renderer.hpp>

#include <common/version.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QIODevice>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtCore/QString>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QSGRendererInterface>

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(exampleRendererCategory)
Q_LOGGING_CATEGORY(exampleRendererCategory, "viewer.renderer")
}  // namespace

Renderer::Renderer(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr, vk::Device device, uint32_t queueFamilyIndex, vk::Queue queue)
    : vkGetInstanceProcAddr{vkGetInstanceProcAddr}, instance{instance->vkInstance()}, physicalDevice{physicalDevice}, vkGetDeviceProcAddr{vkGetDeviceProcAddr}, device{device}, queueFamilyIndex{queueFamilyIndex}, queue{queue}
{
    Q_ASSERT(instance && instance->isValid());
    Q_ASSERT(physicalDevice);
    Q_ASSERT(device);

    instanceFunctions = instance->functions();
    Q_ASSERT(instanceFunctions);
    deviceFunctions = instance->deviceFunctions(device);
    Q_ASSERT(deviceFunctions);

    if (vertexShader.isEmpty()) {
        prepareShader(Stage::Vertex);
        Q_ASSERT(!vertexShader.isEmpty());
    }
    if (fragmentShader.isEmpty()) {
        prepareShader(Stage::Fragment);
        Q_ASSERT(!fragmentShader.isEmpty());
    }
}

Renderer::~Renderer()
{
    qDebug("cleanup");
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

    qDebug("released");
}

void Renderer::setT(float t)
{
    this->t = t;
}

void Renderer::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    if (!pipelineLayoutsAndDescriptorsInitialized) {
        pipelineLayoutsAndDescriptorsInitialized = true;
        initPipelineLayouts(graphicsStateInfo.framesInFlight);
        initDescriptors();
    }

    VkDeviceSize ubufOffset = graphicsStateInfo.currentFrameSlot * uniformBufferPerFrameSize;
    void * p = nullptr;
    VkResult err = deviceFunctions->vkMapMemory(device, uniformBufferMemory, ubufOffset, uniformBufferPerFrameSize, 0, &p);
    Q_CHECK_PTR(p);
    if (err != VK_SUCCESS || !p) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to map uniform buffer memory: %d", err);
    memcpy(p, &t, sizeof t);
    deviceFunctions->vkUnmapMemory(device, uniformBufferMemory);
}

static const float vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};

const int kUBufSize = 4;

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, QSizeF size)
{
    if (!pipelinesInitialized) {
        pipelinesInitialized = true;
        initPipelines(renderPass);
    }

    // Must query the command buffer _after_ beginExternalCommands(), this is
    // actually important when running on Vulkan because what we get here is a
    // new secondary command buffer, not the primary one.
    VkCommandBuffer cb = commandBuffer;
    Q_ASSERT(cb);

    // Do not assume any state persists on the command buffer. (it may be a
    // brand new one that just started recording)

    deviceFunctions->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkDeviceSize vbufOffset = 0;
    deviceFunctions->vkCmdBindVertexBuffers(cb, 0, 1, &vertexBuffer, &vbufOffset);

    uint32_t dynamicOffset = uniformBufferPerFrameSize * graphicsStateInfo.currentFrameSlot;
    deviceFunctions->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uniformBufferDescriptorSet, 1, &dynamicOffset);

    VkViewport vp = {0, 0, utils::autoCast(size.width()), utils::autoCast(size.height()), 0.0f, 1.0f};
    deviceFunctions->vkCmdSetViewport(cb, 0, 1, &vp);
    VkRect2D scissor = {{0, 0}, {utils::autoCast(size.width()), utils::autoCast(size.height())}};
    deviceFunctions->vkCmdSetScissor(cb, 0, 1, &scissor);

    deviceFunctions->vkCmdDraw(cb, 4, 1, 0, 0);
}

void Renderer::prepareShader(Stage stage)
{
    const auto kUri = QStringLiteral("SahKdTree");
    QString stageName;
    switch (stage) {
    case Stage::Vertex: {
        stageName = QStringLiteral("vert");
        break;
    }
    case Stage::Fragment: {
        stageName = QStringLiteral("frag");
        break;
    }
    }
    INVARIANT(!stageName.isEmpty(), "Unknown stage {}", fmt::underlying(stage));
    QFile shaderFile{QLatin1String(":/%1/imports/%2/shaders/fullscreen_triangle.%3.spv").arg(QString::fromUtf8(sah_kd_tree::kProjectName), kUri, stageName)};
    if (!shaderFile.open(QIODevice::ReadOnly)) {
        QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to read shader %s", qPrintable(shaderFile.fileName()));
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

void Renderer::initPipelineLayouts(int framesInFlight)
{
    qDebug("init");

    Q_ASSERT(framesInFlight <= 3);

    // For simplicity we just use host visible buffers instead of device local + staging.

    VkPhysicalDeviceProperties physDevProps = {};
    instanceFunctions->vkGetPhysicalDeviceProperties(physicalDevice, &physDevProps);

    VkPhysicalDeviceMemoryProperties physDevMemProps = {};
    instanceFunctions->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physDevMemProps);

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(vertices);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VkResult err = deviceFunctions->vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create vertex buffer: %d", err);

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
    if (memTypeIndex == uint32_t(-1)) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to find host visible and coherent memory type");

    allocInfo.memoryTypeIndex = memTypeIndex;
    err = deviceFunctions->vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate vertex buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    void * p = nullptr;
    err = deviceFunctions->vkMapMemory(device, vertexBufferMemory, 0, allocInfo.allocationSize, 0, &p);
    INVARIANT(p, "Just initialized");
    if (err != VK_SUCCESS || !p) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to map vertex buffer memory: %d", err);
    memcpy(p, vertices, sizeof(vertices));
    deviceFunctions->vkUnmapMemory(device, vertexBufferMemory);
    err = deviceFunctions->vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to bind vertex buffer memory: %d", err);

    uniformBufferPerFrameSize = aligned(kUBufSize, physDevProps.limits.minUniformBufferOffsetAlignment);

    bufferInfo.size = framesInFlight * uniformBufferPerFrameSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    err = deviceFunctions->vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffer);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create uniform buffer: %d", err);
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
    if (memTypeIndex == uint32_t(-1)) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to find host visible and coherent memory type");

    allocInfo.allocationSize = framesInFlight * uniformBufferPerFrameSize;
    allocInfo.memoryTypeIndex = memTypeIndex;
    err = deviceFunctions->vkAllocateMemory(device, &allocInfo, nullptr, &uniformBufferMemory);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate uniform buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    err = deviceFunctions->vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to bind uniform buffer memory: %d", err);

    // Now onto the pipeline.

    VkPipelineCacheCreateInfo pipelineCacheInfo = {};
    pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    err = deviceFunctions->vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create pipeline cache: %d", err);

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
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create descriptor set layout: %d", err);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    err = deviceFunctions->vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    if (err != VK_SUCCESS) qWarning("Failed to create pipeline layout: %d", err);
}

void Renderer::initDescriptors()
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
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create descriptor pool: %d", err);

    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptorSetLayout;
    err = deviceFunctions->vkAllocateDescriptorSets(device, &descAllocInfo, &uniformBufferDescriptorSet);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate descriptor set");

    VkWriteDescriptorSet writeInfo = {};
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstSet = uniformBufferDescriptorSet;
    writeInfo.dstBinding = 0;
    writeInfo.descriptorCount = 1;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = uniformBuffer;
    bufInfo.offset = 0;  // dynamic offset is used so this is ignored
    bufInfo.range = kUBufSize;
    writeInfo.pBufferInfo = &bufInfo;
    deviceFunctions->vkUpdateDescriptorSets(device, 1, &writeInfo, 0, nullptr);
}

void Renderer::initPipelines(vk::RenderPass renderPass)
{
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = std::size(vertexShader);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(vertexShader.constData());
    VkShaderModule vertShaderModule = {};
    VkResult err = deviceFunctions->vkCreateShaderModule(device, &shaderInfo, nullptr, &vertShaderModule);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create vertex shader module: %d", err);

    shaderInfo.codeSize = std::size(fragmentShader);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(fragmentShader.constData());
    VkShaderModule fragShaderModule = {};
    err = deviceFunctions->vkCreateShaderModule(device, &shaderInfo, nullptr, &fragShaderModule);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create fragment shader module: %d", err);

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

    VkVertexInputBindingDescription vertexBinding = {0,                  // binding
                                                     2 * sizeof(float),  // stride
                                                     VK_VERTEX_INPUT_RATE_VERTEX};
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

    // SrcAlpha, One
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

    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create graphics pipeline: %d", err);
}

}  // namespace viewer
