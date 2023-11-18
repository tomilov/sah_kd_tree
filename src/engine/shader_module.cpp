#include <codegen/vulkan_utils.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/file_io.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/shader_module.hpp>
#include <engine/spirv_reflect_dump.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>

#include <../SPIRV-Reflect/spirv_reflect.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <string_view>
#include <vector>

#include <cstdint>

namespace engine
{

namespace
{

[[nodiscard]] const char * spvReflectResultToString(SpvReflectResult result)
{
    switch (result) {
    case SPV_REFLECT_RESULT_SUCCESS:
        return "SUCCESS";
    case SPV_REFLECT_RESULT_NOT_READY:
        return "NOT_READY";
    case SPV_REFLECT_RESULT_ERROR_PARSE_FAILED:
        return "ERROR_PARSE_FAILED";
    case SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED:
        return "ERROR_ALLOC_FAILED";
    case SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED:
        return "ERROR_RANGE_EXCEEDED";
    case SPV_REFLECT_RESULT_ERROR_NULL_POINTER:
        return "ERROR_NULL_POINTER";
    case SPV_REFLECT_RESULT_ERROR_INTERNAL_ERROR:
        return "ERROR_INTERNAL_ERROR";
    case SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH:
        return "ERROR_COUNT_MISMATCH";
    case SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND:
        return "ERROR_ELEMENT_NOT_FOUND";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE:
        return "ERROR_SPIRV_INVALID_CODE_SIZE";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_MAGIC_NUMBER:
        return "ERROR_SPIRV_INVALID_MAGIC_NUMBER";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF:
        return "ERROR_SPIRV_UNEXPECTED_EOF";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE:
        return "ERROR_SPIRV_INVALID_ID_REFERENCE";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_SET_NUMBER_OVERFLOW:
        return "ERROR_SPIRV_SET_NUMBER_OVERFLOW";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_STORAGE_CLASS:
        return "ERROR_SPIRV_INVALID_STORAGE_CLASS";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_RECURSION:
        return "ERROR_SPIRV_RECURSION";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_INSTRUCTION:
        return "ERROR_SPIRV_INVALID_INSTRUCTION";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_BLOCK_DATA:
        return "ERROR_SPIRV_UNEXPECTED_BLOCK_DATA";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE:
        return "ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ENTRY_POINT:
        return "ERROR_SPIRV_INVALID_ENTRY_POINT";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_EXECUTION_MODE:
        return "ERROR_SPIRV_INVALID_EXECUTION_MODE";
    case SPV_REFLECT_RESULT_ERROR_SPIRV_MAX_RECURSIVE_EXCEEDED:
        return "ERROR_SPIRV_MAX_RECURSIVE_EXCEEDED";
    }
    INVARIANT(false, "Unknown SpvReflectResult value {}", fmt::underlying(result));
}

}  // namespace

}  // namespace engine

template<>
struct fmt::formatter<SpvReflectResult> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectResult reflectResult, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(engine::spvReflectResultToString(reflectResult), ctx);
    }
};

namespace engine
{

namespace
{

[[nodiscard]] vk::ShaderStageFlagBits shaderNameToStage(std::string_view shaderName)
{
    using namespace std::string_view_literals;
    if (shaderName.ends_with(".vert")) {
        return vk::ShaderStageFlagBits::eVertex;
    } else if (shaderName.ends_with(".tesc")) {
        return vk::ShaderStageFlagBits::eTessellationControl;
    } else if (shaderName.ends_with(".tese")) {
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    } else if (shaderName.ends_with(".geom")) {
        return vk::ShaderStageFlagBits::eGeometry;
    } else if (shaderName.ends_with(".frag")) {
        return vk::ShaderStageFlagBits::eFragment;
    } else if (shaderName.ends_with(".comp")) {
        return vk::ShaderStageFlagBits::eCompute;
    } else if (shaderName.ends_with(".rgen")) {
        return vk::ShaderStageFlagBits::eRaygenKHR;
    } else if (shaderName.ends_with(".rahit")) {
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    } else if (shaderName.ends_with(".rchit")) {
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    } else if (shaderName.ends_with(".rmiss")) {
        return vk::ShaderStageFlagBits::eMissKHR;
    } else if (shaderName.ends_with(".rint")) {
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    } else if (shaderName.ends_with(".rcall")) {
        return vk::ShaderStageFlagBits::eCallableKHR;
    } else if (shaderName.ends_with(".task")) {
        return vk::ShaderStageFlagBits::eTaskEXT;
    } else if (shaderName.ends_with(".mesh")) {
        return vk::ShaderStageFlagBits::eMeshEXT;
    } else {
        INVARIANT(false, "Cannot infer stage from shader name '{}'", shaderName);
    }
}

[[nodiscard]] const char * shaderStageToName [[maybe_unused]] (vk::ShaderStageFlagBits shaderStage)
{
    switch (shaderStage) {
    case vk::ShaderStageFlagBits::eVertex:
        return "vert";
    case vk::ShaderStageFlagBits::eTessellationControl:
        return "tesc";
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return "tese";
    case vk::ShaderStageFlagBits::eGeometry:
        return "geom";
    case vk::ShaderStageFlagBits::eFragment:
        return "frag";
    case vk::ShaderStageFlagBits::eCompute:
        return "comp";
    case vk::ShaderStageFlagBits::eAllGraphics:
        return nullptr;
    case vk::ShaderStageFlagBits::eAll:
        return nullptr;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return "rgen";
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return "rahit";
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return "rchit";
    case vk::ShaderStageFlagBits::eMissKHR:
        return "rmiss";
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return "rint";
    case vk::ShaderStageFlagBits::eCallableKHR:
        return "rcall";
    case vk::ShaderStageFlagBits::eTaskEXT:
        return "task";
    case vk::ShaderStageFlagBits::eMeshEXT:
        return "mesh";
    case vk::ShaderStageFlagBits::eSubpassShadingHUAWEI:
        return nullptr;
    case vk::ShaderStageFlagBits::eClusterCullingHUAWEI:
        return nullptr;
    }
    INVARIANT(false, "Unknown shader stage {}", fmt::underlying(shaderStage));
}

[[nodiscard]] SpvReflectShaderStageFlagBits vkShaderStageToSpvReflect [[maybe_unused]] (vk::ShaderStageFlagBits shaderStageFlagBits)
{
    switch (shaderStageFlagBits) {
    case vk::ShaderStageFlagBits::eVertex:
        return SPV_REFLECT_SHADER_STAGE_VERTEX_BIT;
    case vk::ShaderStageFlagBits::eTessellationControl:
        return SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case vk::ShaderStageFlagBits::eGeometry:
        return SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT;
    case vk::ShaderStageFlagBits::eFragment:
        return SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT;
    case vk::ShaderStageFlagBits::eCompute:
        return SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT;
    case vk::ShaderStageFlagBits::eTaskEXT:
        return SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV;
    case vk::ShaderStageFlagBits::eMeshEXT:
        return SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case vk::ShaderStageFlagBits::eMissKHR:
        return SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case vk::ShaderStageFlagBits::eCallableKHR:
        return SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR;
    case vk::ShaderStageFlagBits::eAll:
    case vk::ShaderStageFlagBits::eAllGraphics:
    case vk::ShaderStageFlagBits::eSubpassShadingHUAWEI:
    case vk::ShaderStageFlagBits::eClusterCullingHUAWEI: {
        INVARIANT(false, "Shader stage flag {} is not handled", shaderStageFlagBits);
        break;
    }
    }
    INVARIANT(false, "Shader stage {} is unknown", fmt::underlying(shaderStageFlagBits));
}

[[nodiscard]] vk::DescriptorType spvReflectDescriiptorTypeToVk(SpvReflectDescriptorType descriptorType)
{
    switch (descriptorType) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        return vk::DescriptorType::eSampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return vk::DescriptorType::eCombinedImageSampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return vk::DescriptorType::eSampledImage;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return vk::DescriptorType::eStorageImage;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return vk::DescriptorType::eUniformTexelBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return vk::DescriptorType::eStorageTexelBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return vk::DescriptorType::eUniformBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return vk::DescriptorType::eStorageBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return vk::DescriptorType::eUniformBufferDynamic;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return vk::DescriptorType::eStorageBufferDynamic;
    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return vk::DescriptorType::eInputAttachment;
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return vk::DescriptorType::eAccelerationStructureKHR;
    }
    INVARIANT(false, "Unknown spv descriptor type {}", fmt::underlying(descriptorType));
}

[[nodiscard]] SpvReflectDescriptorType vkDescriptorTypeToSpvReflect [[maybe_unused]] (vk::DescriptorType descriptorType)
{
    switch (descriptorType) {
    case vk::DescriptorType::eSampler:
        return SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER;
    case vk::DescriptorType::eCombinedImageSampler:
        return SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case vk::DescriptorType::eSampledImage:
        return SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case vk::DescriptorType::eStorageImage:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case vk::DescriptorType::eUniformTexelBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case vk::DescriptorType::eStorageTexelBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case vk::DescriptorType::eUniformBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case vk::DescriptorType::eStorageBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case vk::DescriptorType::eUniformBufferDynamic:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    case vk::DescriptorType::eStorageBufferDynamic:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    case vk::DescriptorType::eInputAttachment:
        return SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    case vk::DescriptorType::eAccelerationStructureKHR:
        return SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case vk::DescriptorType::eInlineUniformBlock:
    case vk::DescriptorType::eAccelerationStructureNV:
    case vk::DescriptorType::eMutableEXT:
    case vk::DescriptorType::eSampleWeightImageQCOM:
    case vk::DescriptorType::eBlockMatchImageQCOM: {
        INVARIANT(false, "Descriptor type {} is not handled", descriptorType);
        break;
    }
    }
    INVARIANT(false, "Descriptor type {} is unknown", fmt::underlying(descriptorType));
}

[[nodiscard]] vk::ShaderStageFlagBits spvReflectShaderStageToVk(SpvReflectShaderStageFlagBits shaderStageFlagBits)
{
    switch (shaderStageFlagBits) {
    case SPV_REFLECT_SHADER_STAGE_VERTEX_BIT:
        return vk::ShaderStageFlagBits::eVertex;
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        return vk::ShaderStageFlagBits::eTessellationControl;
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT:
        return vk::ShaderStageFlagBits::eGeometry;
    case SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT:
        return vk::ShaderStageFlagBits::eFragment;
    case SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT:
        return vk::ShaderStageFlagBits::eCompute;
    case SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV:
        return vk::ShaderStageFlagBits::eTaskEXT;
    case SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV:
        return vk::ShaderStageFlagBits::eMeshEXT;
    case SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR:
        return vk::ShaderStageFlagBits::eRaygenKHR;
    case SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR:
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    case SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    case SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR:
        return vk::ShaderStageFlagBits::eMissKHR;
    case SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR:
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    case SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR:
        return vk::ShaderStageFlagBits::eCallableKHR;
    }
    INVARIANT(false, "Unknown spv shader stage {}", fmt::underlying(shaderStageFlagBits));
}

}  // namespace

ShaderModule::ShaderModule(std::string_view name, const Engine & engine, const FileIo & fileIo) : name{name}, engine{engine}, fileIo{fileIo}, library{engine.getLibrary()}, device{engine.getDevice()}
{
    load();
}

void ShaderModule::load()
{
    shaderStage = shaderNameToStage(name);
    spirv = fileIo.loadShader(name);

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo.setCode(spirv);
    shaderModuleHolder = device.device.createShaderModuleUnique(shaderModuleCreateInfo, library.allocationCallbacks, library.dispatcher);
    shaderModule = *shaderModuleHolder;

    device.setDebugUtilsObjectName(shaderModule, name);
}

ShaderModuleReflection::ShaderModuleReflection(const ShaderModule & shaderModule, std::string_view entryPoint) : shaderModule{shaderModule}, entryPoint{entryPoint}, reflectionModule{shaderModule.spirv, SPV_REFLECT_MODULE_FLAG_NO_COPY}
{
    auto reflectionResult = reflectionModule->GetResult();
    INVARIANT(reflectionResult == SPV_REFLECT_RESULT_SUCCESS, "spvReflectCreateShaderModule returned {} for shader module '{}'", reflectionResult, shaderModule.name);

    dump(*reflectionModule);

    reflect();
}

ShaderModuleReflection::~ShaderModuleReflection() = default;

VertexInputState ShaderModuleReflection::getVertexInputState(uint32_t vertexBufferBinding) const
{
    INVARIANT(shaderStage == vk::ShaderStageFlagBits::eVertex, "Pipeline vertex input state can be only inferred for vertex shader, not {}", shaderStage);

    SpvReflectResult reflectResult = SPV_REFLECT_RESULT_SUCCESS;
    VertexInputState vertexInputState;

    auto & vertexInputBindingDescriptions = vertexInputState.vertexInputBindingDescriptions;
    auto & vertexInputBindingDescription = vertexInputBindingDescriptions.emplace_back();
    vertexInputBindingDescription.binding = vertexBufferBinding;
    vertexInputBindingDescription.stride = 0;
    vertexInputBindingDescription.inputRate = vk::VertexInputRate::eVertex;

    uint32_t inputVariableCount = 0;
    reflectResult = reflectionModule->EnumerateEntryPointInputVariables(entryPoint.c_str(), &inputVariableCount, nullptr);
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateInputVariables returned {}", reflectResult);
    std::vector<SpvReflectInterfaceVariable *> reflectInterfaceVariable(inputVariableCount);
    reflectResult = reflectionModule->EnumerateEntryPointInputVariables(entryPoint.c_str(), &inputVariableCount, std::data(reflectInterfaceVariable));
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateInputVariables returned {}", reflectResult);

    const auto locationLess = [](auto l, auto r) -> bool
    {
        INVARIANT(l && r, "");
        return l->location < r->location;
    };
    std::sort(std::begin(reflectInterfaceVariable), std::end(reflectInterfaceVariable), locationLess);

    auto & variableNames = vertexInputState.variableNames;
    auto & vertexInputAttributeDescriptions = vertexInputState.vertexInputAttributeDescriptions;
    for (const auto inputVariable : reflectInterfaceVariable) {
        INVARIANT(inputVariable, "");
        auto variableName = inputVariable->name ? inputVariable->name : fmt::to_string(inputVariable->spirv_id);
        SPDLOG_DEBUG("Variable name: '{}'", variableName);
        variableNames.push_back(std::move(variableName));
        if (inputVariable->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN) {
            continue;
        }
        auto & vertexInputAttributeDescription = vertexInputAttributeDescriptions.emplace_back();
        vertexInputAttributeDescription.location = inputVariable->location;
        vertexInputAttributeDescription.binding = vertexInputBindingDescription.binding;
        vertexInputAttributeDescription.format = utils::autoCast(inputVariable->format);
        vertexInputAttributeDescription.offset = vertexInputBindingDescription.stride;

        auto formatProperties = shaderModule.device.physicalDevice.physicalDevice.getFormatProperties(vertexInputAttributeDescription.format, shaderModule.library.dispatcher);
        INVARIANT(formatProperties.bufferFeatures & vk::FormatFeatureFlagBits::eVertexBuffer, "");

        auto formatSize = codegen::vulkan::formatElementSize(vertexInputAttributeDescription.format, vk::ImageAspectFlagBits::eColor);
        INVARIANT(formatSize > 0, "Expected known to VkLayer_utils format {}", vertexInputAttributeDescription.format);
        vertexInputBindingDescription.stride += formatSize;
    }

    auto & pipelineVertexInputStateCreateInfo = vertexInputState.pipelineVertexInputStateCreateInfo;
    pipelineVertexInputStateCreateInfo.flags = {};
    pipelineVertexInputStateCreateInfo.setVertexAttributeDescriptions(vertexInputAttributeDescriptions);
    pipelineVertexInputStateCreateInfo.setVertexBindingDescriptions(vertexInputBindingDescriptions);
    return vertexInputState;
}

void ShaderModuleReflection::reflect()
{
    SpvReflectResult reflectResult = SPV_REFLECT_RESULT_SUCCESS;

    bool entryPointFound = false;
    auto entryPointCount = reflectionModule->GetEntryPointCount();
    SPDLOG_DEBUG("Shader consists of {} entry points", entryPointCount);
    for (uint32_t i = 0; i < entryPointCount; ++i) {
        auto entryPointName = reflectionModule->GetEntryPointName(i);
        if (entryPointName == entryPoint) {
            SPDLOG_DEBUG("Found entry point '{}'", entryPointName);
            shaderStage = spvReflectShaderStageToVk(reflectionModule->GetEntryPointShaderStage(i));
            entryPointFound = true;
            break;
        }
    }
    INVARIANT(entryPointFound, "Entry point '{}' is not found", entryPoint);
    if (!(shaderStage & shaderModule.shaderStage)) {
        SPDLOG_WARN("Reflected flags ({}) of shader module '{}' does not contain inferred flags ({})", shaderStage, shaderModule.name, shaderModule.shaderStage);
    }

    uint32_t descriptorSetCount = 0;
    reflectResult = reflectionModule->EnumerateEntryPointDescriptorSets(entryPoint.c_str(), &descriptorSetCount, nullptr);
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);
    std::vector<SpvReflectDescriptorSet *> reflectDescriptorSets(descriptorSetCount);
    reflectResult = reflectionModule->EnumerateEntryPointDescriptorSets(entryPoint.c_str(), &descriptorSetCount, std::data(reflectDescriptorSets));
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);

    for (uint32_t index = 0; index < descriptorSetCount; ++index) {
        const auto reflectDecriptorSet = reflectDescriptorSets.at(index);
        INVARIANT(reflectDecriptorSet, "reflectDecriptorSet is null at #{}", index);
        INVARIANT(!descriptorSetLayoutSetBindings.contains(reflectDecriptorSet->set), "Duplicated set {}", reflectDecriptorSet->set);
        auto & descriptorSetLayoutBindings = descriptorSetLayoutSetBindings[reflectDecriptorSet->set];
        auto bindingCount = reflectDecriptorSet->binding_count;
        descriptorSetLayoutBindings.reserve(bindingCount);
        for (uint32_t b = 0; b < bindingCount; ++b) {
            const auto reflectDescriptorBinding = reflectDecriptorSet->bindings[b];
            INVARIANT(reflectDescriptorBinding, "");
            auto descriptorBindingName = reflectDescriptorBinding->name ? reflectDescriptorBinding->name : fmt::to_string(reflectDescriptorBinding->spirv_id);
            INVARIANT(!descriptorSetLayoutBindings.contains(descriptorBindingName), "Duplicated descriptor binding name '{}'", descriptorBindingName);
            auto & descriptorSetLayoutBinding = descriptorSetLayoutBindings[descriptorBindingName];
            descriptorSetLayoutBinding.binding = reflectDescriptorBinding->binding;
            descriptorSetLayoutBinding.descriptorType = spvReflectDescriiptorTypeToVk(reflectDescriptorBinding->descriptor_type);
            descriptorSetLayoutBinding.descriptorCount = reflectDescriptorBinding->count;
            for (uint32_t d = 0; d < reflectDescriptorBinding->array.dims_count; ++d) {
                descriptorSetLayoutBinding.descriptorCount *= reflectDescriptorBinding->array.dims[d];
            }
            descriptorSetLayoutBinding.stageFlags = shaderStage;
        }
    }

    uint32_t pushConstantBlockCount = 0;
    reflectResult = reflectionModule->EnumerateEntryPointPushConstantBlocks(entryPoint.c_str(), &pushConstantBlockCount, nullptr);
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumeratePushConstantBlocks returned {}", reflectResult);
    std::vector<SpvReflectBlockVariable *> pushConstantBlocks(pushConstantBlockCount);
    reflectResult = reflectionModule->EnumerateEntryPointPushConstantBlocks(entryPoint.c_str(), &pushConstantBlockCount, std::data(pushConstantBlocks));
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumeratePushConstantBlocks returned {}", reflectResult);

    if (pushConstantBlockCount > 0) {
        auto & [stageFlags, offset, size] = pushConstantRange.emplace();
        stageFlags = shaderStage;
        offset = pushConstantBlocks.front()->offset;
        size = pushConstantBlocks.front()->size;
        for (uint32_t index = 1; index < pushConstantBlockCount; ++index) {
            const auto reflectPushConstantBlock = pushConstantBlocks.at(index);
            INVARIANT(reflectPushConstantBlock, "");
            if (offset > reflectPushConstantBlock->offset) {
                offset = reflectPushConstantBlock->offset;
            }
            if (offset + size < reflectPushConstantBlock->offset + reflectPushConstantBlock->size) {
                size = reflectPushConstantBlock->offset - offset + reflectPushConstantBlock->size;
            }
        }
    }
}

ShaderStages::ShaderStages(const Engine & engine, uint32_t vertexBufferBinding) : engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, vertexBufferBinding{vertexBufferBinding}
{}

void ShaderStages::append(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection, std::string_view entryPoint)
{
    shaderModuleReflections.emplace_back(shaderModuleReflection);
    entryPoints.emplace_back(entryPoint);
    const auto & name = names.emplace_back(fmt::format("{}:{}", shaderModule.name, entryPoint));

    shaderStages.emplace_back();
    auto & pipelineShaderStageCreateInfo = shaderStages.back<vk::PipelineShaderStageCreateInfo>();
    pipelineShaderStageCreateInfo = {
        .flags = {},
        .stage = shaderModule.shaderStage,
        .module = shaderModule.shaderModule,
        .pName = entryPoints.back().c_str(),
        .pSpecializationInfo = nullptr,
    };
    auto & debugUtilsObjectNameInfo = shaderStages.back<vk::DebugUtilsObjectNameInfoEXT>();
    debugUtilsObjectNameInfo.objectType = shaderModule.shaderModule.objectType;
    debugUtilsObjectNameInfo.objectHandle = utils::autoCast(typename vk::ShaderModule::NativeType(shaderModule.shaderModule));
    debugUtilsObjectNameInfo.pObjectName = name.c_str();

    if (shaderModule.shaderStage == vk::ShaderStageFlagBits::eVertex) {
        vertexInputState = shaderModuleReflection.getVertexInputState(vertexBufferBinding);
    }

    for (const auto & [set, bindings] : shaderModuleReflection.descriptorSetLayoutSetBindings) {
        auto & mergedBindings = setBindings[set];
        for (const auto & [bindingName, binding] : bindings) {
            bool merged = false;
            size_t b = 0;
            for (auto & mergedBinding : mergedBindings.bindings) {
                if (binding.binding == mergedBinding.binding) {
                    INVARIANT(binding.descriptorType == mergedBinding.descriptorType, "{} != {} (binding {})", binding.descriptorType, mergedBinding.descriptorType, b);
                    INVARIANT(binding.descriptorCount == mergedBinding.descriptorCount, "{} != {} (binding {})", binding.descriptorCount, mergedBinding.descriptorCount, b);
                    INVARIANT(binding.pImmutableSamplers == mergedBinding.pImmutableSamplers, "{} != {} (binding {})", fmt::ptr(binding.pImmutableSamplers), fmt::ptr(mergedBinding.pImmutableSamplers), b);
                    mergedBinding.stageFlags |= binding.stageFlags;
                    merged = true;
                    break;
                }
                ++b;
            }
            if (!merged) {
                size_t index = std::size(mergedBindings.bindings);
                mergedBindings.bindings.push_back(binding);
                mergedBindings.bindingIndices.emplace(bindingName, index);
                mergedBindings.bindingNames.push_back(bindingName);
            }
        }
    }
    uint32_t setIndex = 0;
    for (auto & [set, bindings] : setBindings) {
        bindings.setIndex = setIndex++;
    }

    if (shaderModuleReflection.pushConstantRange) {
        pushConstantRanges.push_back(shaderModuleReflection.pushConstantRange.value());
    }
}

void ShaderStages::createDescriptorSetLayouts(std::string_view name, vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags)
{
    size_t setCount = std::size(setBindings);
    descriptorSetLayoutHolders.reserve(setCount);
    descriptorSetLayouts.reserve(setCount);

    for (const auto & [set, descriptorSetLayoutBindings] : setBindings) {
        vk::StructureChain<vk::DescriptorSetLayoutCreateInfo, vk::DescriptorSetLayoutBindingFlagsCreateInfo> descriptorSetLayoutCreateInfoChain;
        auto & descriptorSetLayoutCreateInfo = descriptorSetLayoutCreateInfoChain.get<vk::DescriptorSetLayoutCreateInfo>();
        descriptorSetLayoutCreateInfo.flags = descriptorSetLayoutCreateFlags;
        descriptorSetLayoutCreateInfo.setBindings(descriptorSetLayoutBindings.bindings);
        auto & descriptorSetLayoutBindingFlagsCreateInfo = descriptorSetLayoutCreateInfoChain.get<vk::DescriptorSetLayoutBindingFlagsCreateInfo>();
        descriptorSetLayoutBindingFlagsCreateInfo.setBindingFlags(nullptr);  // TODO:
        descriptorSetLayoutHolders.push_back(device.device.createDescriptorSetLayoutUnique(descriptorSetLayoutCreateInfo, library.allocationCallbacks, library.dispatcher));
        descriptorSetLayouts.push_back(*descriptorSetLayoutHolders.back());

        for (const auto & descriptorSetLayoutBinding : descriptorSetLayoutBindings.bindings) {
            if (descriptorSetLayoutCreateFlags & vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT) {
                INVARIANT(descriptorSetLayoutBinding.descriptorType != vk::DescriptorType::eUniformBufferDynamic, "Not compatible with eDescriptorBufferEXT descriptor set layout");
                INVARIANT(descriptorSetLayoutBinding.descriptorType != vk::DescriptorType::eStorageBufferDynamic, "Not compatible with eDescriptorBufferEXT descriptor set layout");
            } else {
                descriptorCounts[descriptorSetLayoutBinding.descriptorType] += descriptorSetLayoutBinding.descriptorCount;
            }
        }

        if (std::size(setBindings) > 1) {
            auto descriptorSetLayoutName = fmt::format("{} set {} (of total {} sets)", name, set, setCount);
            device.setDebugUtilsObjectName(descriptorSetLayouts.back(), descriptorSetLayoutName);
        } else {
            auto descriptorSetLayoutName = fmt::format("{} set {}", name, set);
            device.setDebugUtilsObjectName(descriptorSetLayouts.back(), descriptorSetLayoutName);
        }
    }
}

}  // namespace engine
