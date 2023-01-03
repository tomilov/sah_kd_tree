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

Renderer::Renderer(GetInstanceProcAddress getInstanceProcAddress, QVulkanInstance * instance, vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, vk::Queue queue)
    : getInstanceProcAddress{getInstanceProcAddress}, instance{instance}, physicalDevice{physicalDevice}, device{device}, queueFamilyIndex{queueFamilyIndex}, queue{queue}
{
    Q_ASSERT(instance && instance->isValid());

    Q_ASSERT(physicalDevice && device);

    m_devFuncs = instance->deviceFunctions(device);
    m_funcs = instance->functions();
    Q_ASSERT(m_devFuncs && m_funcs);
}

Renderer::~Renderer()
{
    qDebug("cleanup");
    if (!m_devFuncs) return;

    m_devFuncs->vkDestroyPipeline(device, m_pipeline, nullptr);
    m_devFuncs->vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    m_devFuncs->vkDestroyDescriptorSetLayout(device, m_resLayout, nullptr);

    m_devFuncs->vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);

    m_devFuncs->vkDestroyPipelineCache(device, m_pipelineCache, nullptr);

    m_devFuncs->vkDestroyBuffer(device, m_vbuf, nullptr);
    m_devFuncs->vkFreeMemory(device, m_vbufMem, nullptr);

    m_devFuncs->vkDestroyBuffer(device, m_ubuf, nullptr);
    m_devFuncs->vkFreeMemory(device, m_ubufMem, nullptr);

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

    VkDeviceSize ubufOffset = graphicsStateInfo.currentFrameSlot * m_allocPerUbuf;
    void * p = nullptr;
    VkResult err = m_devFuncs->vkMapMemory(device, m_ubufMem, ubufOffset, m_allocPerUbuf, 0, &p);
    Q_CHECK_PTR(p);
    if (err != VK_SUCCESS || !p) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to map uniform buffer memory: %d", err);
    memcpy(p, &t, sizeof t);
    m_devFuncs->vkUnmapMemory(device, m_ubufMem);
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

    m_devFuncs->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    VkDeviceSize vbufOffset = 0;
    m_devFuncs->vkCmdBindVertexBuffers(cb, 0, 1, &m_vbuf, &vbufOffset);

    uint32_t dynamicOffset = m_allocPerUbuf * graphicsStateInfo.currentFrameSlot;
    m_devFuncs->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_ubufDescriptor, 1, &dynamicOffset);

    VkViewport vp = {0, 0, utils::autoCast(size.width()), utils::autoCast(size.height()), 0.0f, 1.0f};
    m_devFuncs->vkCmdSetViewport(cb, 0, 1, &vp);
    VkRect2D scissor = {{0, 0}, {utils::autoCast(size.width()), utils::autoCast(size.height())}};
    m_devFuncs->vkCmdSetScissor(cb, 0, 1, &scissor);

    m_devFuncs->vkCmdDraw(cb, 4, 1, 0, 0);
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
        m_vert = shaderFile.readAll();
        break;
    }
    case Stage::Fragment: {
        m_frag = shaderFile.readAll();
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
    m_funcs->vkGetPhysicalDeviceProperties(physicalDevice, &physDevProps);

    VkPhysicalDeviceMemoryProperties physDevMemProps = {};
    m_funcs->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physDevMemProps);

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(vertices);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VkResult err = m_devFuncs->vkCreateBuffer(device, &bufferInfo, nullptr, &m_vbuf);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create vertex buffer: %d", err);

    VkMemoryRequirements memReq = {};
    m_devFuncs->vkGetBufferMemoryRequirements(device, m_vbuf, &memReq);
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
    err = m_devFuncs->vkAllocateMemory(device, &allocInfo, nullptr, &m_vbufMem);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate vertex buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    void * p = nullptr;
    err = m_devFuncs->vkMapMemory(device, m_vbufMem, 0, allocInfo.allocationSize, 0, &p);
    if (err != VK_SUCCESS || !p) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to map vertex buffer memory: %d", err);
    memcpy(p, vertices, sizeof(vertices));
    m_devFuncs->vkUnmapMemory(device, m_vbufMem);
    err = m_devFuncs->vkBindBufferMemory(device, m_vbuf, m_vbufMem, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to bind vertex buffer memory: %d", err);

    m_allocPerUbuf = aligned(kUBufSize, physDevProps.limits.minUniformBufferOffsetAlignment);

    bufferInfo.size = framesInFlight * m_allocPerUbuf;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    err = m_devFuncs->vkCreateBuffer(device, &bufferInfo, nullptr, &m_ubuf);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create uniform buffer: %d", err);
    m_devFuncs->vkGetBufferMemoryRequirements(device, m_ubuf, &memReq);
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

    allocInfo.allocationSize = framesInFlight * m_allocPerUbuf;
    allocInfo.memoryTypeIndex = memTypeIndex;
    err = m_devFuncs->vkAllocateMemory(device, &allocInfo, nullptr, &m_ubufMem);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate uniform buffer memory of size %u: %d", uint(allocInfo.allocationSize), err);

    err = m_devFuncs->vkBindBufferMemory(device, m_ubuf, m_ubufMem, 0);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to bind uniform buffer memory: %d", err);

    // Now onto the pipeline.

    VkPipelineCacheCreateInfo pipelineCacheInfo = {};
    pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    err = m_devFuncs->vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &m_pipelineCache);
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
    err = m_devFuncs->vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_resLayout);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create descriptor set layout: %d", err);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_resLayout;
    err = m_devFuncs->vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
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
    VkResult err = m_devFuncs->vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &m_descriptorPool);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create descriptor pool: %d", err);

    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = m_descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &m_resLayout;
    err = m_devFuncs->vkAllocateDescriptorSets(device, &descAllocInfo, &m_ubufDescriptor);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to allocate descriptor set");

    VkWriteDescriptorSet writeInfo = {};
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstSet = m_ubufDescriptor;
    writeInfo.dstBinding = 0;
    writeInfo.descriptorCount = 1;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = m_ubuf;
    bufInfo.offset = 0;  // dynamic offset is used so this is ignored
    bufInfo.range = kUBufSize;
    writeInfo.pBufferInfo = &bufInfo;
    m_devFuncs->vkUpdateDescriptorSets(device, 1, &writeInfo, 0, nullptr);
}

void Renderer::initPipelines(vk::RenderPass renderPass)
{
    if (m_vert.isEmpty()) {
        prepareShader(Stage::Vertex);
        Q_ASSERT(!m_vert.isEmpty());
    }
    if (m_frag.isEmpty()) {
        prepareShader(Stage::Fragment);
        Q_ASSERT(!m_frag.isEmpty());
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = std::size(m_vert);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(m_vert.constData());
    VkShaderModule vertShaderModule = {};
    VkResult err = m_devFuncs->vkCreateShaderModule(device, &shaderInfo, nullptr, &vertShaderModule);
    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create vertex shader module: %d", err);

    shaderInfo.codeSize = std::size(m_frag);
    shaderInfo.pCode = reinterpret_cast<const quint32 *>(m_frag.constData());
    VkShaderModule fragShaderModule = {};
    err = m_devFuncs->vkCreateShaderModule(device, &shaderInfo, nullptr, &fragShaderModule);
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

    pipelineInfo.layout = m_pipelineLayout;

    pipelineInfo.renderPass = renderPass;

    err = m_devFuncs->vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_pipeline);

    m_devFuncs->vkDestroyShaderModule(device, vertShaderModule, nullptr);
    m_devFuncs->vkDestroyShaderModule(device, fragShaderModule, nullptr);

    if (err != VK_SUCCESS) QT_MESSAGE_LOGGER_COMMON(exampleRendererCategory, QtFatalMsg).fatal("Failed to create graphics pipeline: %d", err);
}

}  // namespace viewer
