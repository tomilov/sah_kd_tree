#include <codegen/vulkan_utils.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/file_io.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/push_constant_ranges.hpp>
#include <engine/shader_module.hpp>
#include <engine/spirv_reflect_dump.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <../SPIRV-Reflect/spirv_reflect.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <span>
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
    case vk::DescriptorType::eTensorARM: {
        INVARIANT(false, "Not implemented");  // TODO:
    }
    case vk::DescriptorType::ePartitionedAccelerationStructureNV: {
        INVARIANT(false, "Not implemented");  // TODO:
    }
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

ShaderModule::ShaderModule(const Context & contextIn, const FileIo & fileIoIn, std::string_view shaderNameIn)
    : context{contextIn}
    , fileIo{fileIoIn}
    , shaderName{shaderNameIn}
{
    shaderStage = shaderNameToStage(shaderName);
    spirv = fileIo.loadShader(shaderName);
    INVARIANT(!std::empty(spirv), "{}", shaderName);

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo.setCode(spirv);
    shaderModuleHolder = context.getDevice().getHandle().createShaderModuleUnique(shaderModuleCreateInfo, context.getLibrary().getAllocationCallbacks(), context.getDispatcher());

    context.getDevice().setDebugUtilsObjectName(*shaderModuleHolder, shaderName);
}

vk::ShaderModule ShaderModule::getHandle() const &
{
    ASSERT(shaderModuleHolder);
    return *shaderModuleHolder;
}

ShaderModule::operator vk::ShaderModule() const &
{
    return getHandle();
}

ShaderModuleReflection::ShaderModuleReflection(const Context & context, const ShaderModule & shaderModule, std::string_view entryPointName)
    : context{context}
    , shaderModuleName{shaderModule.getShaderName()}
    , shaderStage{shaderModule.getStage()}
    , entryPointName{entryPointName}
    , reflectionModule{shaderModule.getSpirv(), SPV_REFLECT_MODULE_FLAG_NO_COPY}
{
    auto reflectionResult = reflectionModule->GetResult();
    INVARIANT(reflectionResult == SPV_REFLECT_RESULT_SUCCESS, "spvReflectCreateShaderModule returned {} for shader module '{}'", reflectionResult, shaderModuleName);

    dump(*reflectionModule);

    reflect();
}

ShaderModuleReflection::ShaderModuleReflection(ShaderModuleReflection &&) noexcept = default;
ShaderModuleReflection::~ShaderModuleReflection() = default;

vk::ShaderStageFlagBits ShaderModuleReflection::getShaderStage() const
{
    return shaderStage;
}

const std::string & ShaderModuleReflection::getEntryPointName() const &
{
    return entryPointName;
}

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

    std::vector<SpvReflectInterfaceVariable *> reflectInterfaceVariable;
    {
        uint32_t inputVariableCount = 0;
        reflectResult = reflectionModule->EnumerateEntryPointInputVariables(entryPointName.c_str(), &inputVariableCount, nullptr);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateInputVariables returned {}", reflectResult);
        reflectInterfaceVariable.resize(inputVariableCount);
        reflectResult = reflectionModule->EnumerateEntryPointInputVariables(entryPointName.c_str(), &inputVariableCount, std::data(reflectInterfaceVariable));
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateInputVariables returned {}", reflectResult);
    }
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

        auto formatProperties = context.getPhysicalDevice().getHandle().getFormatProperties(vertexInputAttributeDescription.format, context.getDispatcher());
        INVARIANT(formatProperties.bufferFeatures & vk::FormatFeatureFlagBits::eVertexBuffer, "");

        auto formatSize = utils::safeCast<uint32_t>(codegen::vulkan::formatElementSize(vertexInputAttributeDescription.format, vk::ImageAspectFlagBits::eColor));
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

    auto entryPointCount = reflectionModule->GetEntryPointCount();
    SPDLOG_DEBUG("Shader consists of {} entry points", entryPointCount);
    vk::ShaderStageFlags shaderStageMask;
    for (uint32_t i = 0; i < entryPointCount; ++i) {
        auto nextEntryPointName = reflectionModule->GetEntryPointName(i);
        if (nextEntryPointName == entryPointName) {
            SPDLOG_DEBUG("Found entry point '{}'", nextEntryPointName);
            shaderStageMask = spvReflectShaderStageToVk(reflectionModule->GetEntryPointShaderStage(i));
            break;
        }
    }
    INVARIANT(shaderStageMask, "Entry point '{}' is not found", entryPointName);
    INVARIANT(shaderStageMask == shaderStage, "Reflected shader stage ({}) of shader module '{}' does not match inferred shader stage ({})", shaderStageMask, shaderModuleName, shaderStage);

    std::vector<SpvReflectDescriptorSet *> reflectDescriptorSets;
    {
        uint32_t descriptorSetCount = 0;
        reflectResult = reflectionModule->EnumerateEntryPointDescriptorSets(entryPointName.c_str(), &descriptorSetCount, nullptr);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);
        reflectDescriptorSets.resize(descriptorSetCount);
        reflectResult = reflectionModule->EnumerateEntryPointDescriptorSets(entryPointName.c_str(), &descriptorSetCount, std::data(reflectDescriptorSets));
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);
    }
    for (auto reflectDecriptorSet : reflectDescriptorSets) {
        INVARIANT(reflectDecriptorSet, "");
        INVARIANT(!descriptorSetLayoutSetBindings.contains(reflectDecriptorSet->set), "Duplicated set {}", reflectDecriptorSet->set);
        auto & descriptorSetLayoutBindings = descriptorSetLayoutSetBindings[reflectDecriptorSet->set];
        auto bindingCount = reflectDecriptorSet->binding_count;
        descriptorSetLayoutBindings.reserve(bindingCount);
        for (uint32_t b = 0; b < bindingCount; ++b) {
            const auto reflectDescriptorBinding = reflectDecriptorSet->bindings[b];
            INVARIANT(reflectDescriptorBinding, "");
            auto descriptorBindingName = reflectDescriptorBinding->name ? reflectDescriptorBinding->name : "";  // fmt::format("_{}", reflectDescriptorBinding->spirv_id)
            const vk::DescriptorType descriptorType = spvReflectDescriiptorTypeToVk(reflectDescriptorBinding->descriptor_type);
            DescriptorBindingNameAndType descriptorBindingNameAndType{descriptorBindingName, descriptorType};
            INVARIANT(!descriptorSetLayoutBindings.contains(descriptorBindingNameAndType), "Duplicated descriptor binding name '{}' and type {}", descriptorBindingName, descriptorType);
            auto & descriptorSetLayoutBinding = descriptorSetLayoutBindings[std::move(descriptorBindingNameAndType)];
            descriptorSetLayoutBinding.binding = {
                .binding = reflectDescriptorBinding->binding,
                .descriptorType = descriptorType,
                .descriptorCount = 1,  // ? reflectDescriptorBinding->count,
            };
            const auto & block = reflectDescriptorBinding->block;
            ASSERT(block.offset == 0);
            ASSERT(block.absolute_offset == 0);
            descriptorSetLayoutBinding.size = block.size;
            for (uint32_t d = 0; d < reflectDescriptorBinding->array.dims_count; ++d) {
                descriptorSetLayoutBinding.binding.descriptorCount *= reflectDescriptorBinding->array.dims[d];
            }
            descriptorSetLayoutBinding.binding.stageFlags = shaderStage;
        }
    }

    std::vector<SpvReflectBlockVariable *> pushConstantBlocks;
    {
        uint32_t pushConstantBlockCount = 0;
        reflectResult = reflectionModule->EnumerateEntryPointPushConstantBlocks(entryPointName.c_str(), &pushConstantBlockCount, nullptr);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumeratePushConstantBlocks returned {}", reflectResult);
        pushConstantBlocks.resize(pushConstantBlockCount);
        reflectResult = reflectionModule->EnumerateEntryPointPushConstantBlocks(entryPointName.c_str(), &pushConstantBlockCount, std::data(pushConstantBlocks));
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumeratePushConstantBlocks returned {}", reflectResult);
    }
    for (auto reflectPushConstantBlock : pushConstantBlocks) {
        INVARIANT(reflectPushConstantBlock, "");
        auto members = reflectPushConstantBlock->members;
        size_t memberCount = utils::autoCast(reflectPushConstantBlock->member_count);
        for (const SpvReflectBlockVariable & member : std::span<const SpvReflectBlockVariable>(members, memberCount)) {
            if ((member.flags & SPV_REFLECT_VARIABLE_FLAGS_UNUSED) != 0) {
                auto memberName = member.name ? member.name : "<unknown>";
                auto blockName = reflectPushConstantBlock->name ? reflectPushConstantBlock->name : "<unknonw>";
                SPDLOG_WARN("Member {} of {} is not statically used in entry point {} on stage {} of shader {}", memberName, blockName, entryPointName, shaderStage, shaderModuleName);
                continue;
            }
            const bool isInitialized = pushConstantRange.has_value();
            auto & [stageFlags, offset, size] = isInitialized ? pushConstantRange.value() : pushConstantRange.emplace();
            if (isInitialized) {
                if (offset > member.offset) {
                    offset = member.offset;
                }
                if (offset + size < member.offset + member.size) {
                    size = member.offset - offset + member.size;
                }
            } else {
                stageFlags = shaderStage;
                offset = member.offset;
                size = member.size;
            }
        }
    }

    std::vector<SpvReflectSpecializationConstant *> specConstants;
    {
        uint32_t specializationConstantCount = 0;
        reflectResult = reflectionModule->EnumerateSpecializationConstants(&specializationConstantCount, nullptr);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateSpecializationConstants returned {}", reflectResult);
        specConstants.resize(specializationConstantCount);
        reflectResult = reflectionModule->EnumerateSpecializationConstants(&specializationConstantCount, std::data(specConstants));
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateSpecializationConstants returned {}", reflectResult);
    }
    for (auto specConstant : specConstants) {
        if (!specConstant->name) {
            continue;
        }
        if (!specializationConstants.emplace(specConstant->name, specConstant->constant_id).second) {
            INVARIANT(false, "{} {} ({})", specConstant->name, specConstant->constant_id, fmt::join(specializationConstants, ", "));
        }
    }
}

ShaderStages::ShaderStages(const Context & context, uint32_t vertexBufferBinding)
    : context{context}
    , vertexBufferBinding{vertexBufferBinding}
{}

bool ShaderStages::checkSubgroupSize(uint32_t subgroupSize, vk::ShaderStageFlagBits shaderStage) const
{
    const auto & subgroupSizeControlProperties = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceVulkan13Properties>();
    if (subgroupSize < subgroupSizeControlProperties.minSubgroupSize) {
        return false;
    }
    if (subgroupSizeControlProperties.maxSubgroupSize < subgroupSize) {
        return false;
    }
    if (!(subgroupSizeControlProperties.requiredSubgroupSizeStages & shaderStage)) {
        return false;
    }
    return true;
}

void ShaderStages::add(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection, std::optional<uint32_t> subgroupSize)
{
    const auto & entryPointName = shaderModuleReflection.getEntryPointName();
    entryPointNames.push_back(entryPointName);
    names.push_back(fmt::format("{}:{}", shaderModule.getShaderName(), entryPointName));
    INVARIANT(std::size(pipelineShaderStageCreateInfoChains) < pipelineShaderStageCreateInfoChains.capacity(), "");
    auto & [pipelineShaderStageCreateInfo, debugUtilsObjectNameInfo, requiredSubgroupSize] = pipelineShaderStageCreateInfoChains.emplace_back();
    pipelineShaderStageCreateInfo.flags = vk::PipelineShaderStageCreateFlags{};
    pipelineShaderStageCreateInfo.stage = shaderModule.getStage();
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = std::data(entryPointNames.back());
    pipelineShaderStageCreateInfo.pSpecializationInfo = nullptr;
    debugUtilsObjectNameInfo.objectType = shaderModule.getHandle().objectType;
    debugUtilsObjectNameInfo.objectHandle = utils::autoCast(utils::safeCast<typename vk::ShaderModule::NativeType>(shaderModule.getHandle()));
    debugUtilsObjectNameInfo.pObjectName = std::data(names.back());
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan13Features>().subgroupSizeControl != vk::False) {
        if (subgroupSize) {
            INVARIANT(checkSubgroupSize(subgroupSize.value(), shaderModule.getStage()), "");
            requiredSubgroupSize.requiredSubgroupSize = subgroupSize.value();
        } else {
            pipelineShaderStageCreateInfoChains.back().unlink<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo>();
        }
    } else {
        if (subgroupSize) {
            SPDLOG_WARN("subgroupSize is set, but v is not enabled");
        }
    }

    pipelineShaderStageCreateInfos.push_back(pipelineShaderStageCreateInfo);

    if (shaderModule.getStage() == vk::ShaderStageFlagBits::eVertex) {
        vertexInputState = std::make_unique<VertexInputState>(shaderModuleReflection.getVertexInputState(vertexBufferBinding));
    }

    for (const auto & [set, bindings] : shaderModuleReflection.descriptorSetLayoutSetBindings) {
        auto & mergedBindings = setBindings[set];
        for (const auto & [bindingName, binding] : bindings) {
            bool isMultistage = false;
            size_t b = 0;
            for (auto & mergedBinding : mergedBindings.bindings) {
                if (binding.binding.binding == mergedBinding.binding) {
                    const auto & [n, t] = mergedBindings.bindingNames.at(b);
                    INVARIANT(binding.binding.descriptorType == mergedBinding.descriptorType, "{} != {} (binding #{}: {}, {})", binding.binding.descriptorType, mergedBinding.descriptorType, b, n, t);
                    INVARIANT(binding.binding.descriptorCount == mergedBinding.descriptorCount, "{} != {} (binding #{}: {}, {})", binding.binding.descriptorCount, mergedBinding.descriptorCount, b, n, t);
                    INVARIANT(binding.binding.pImmutableSamplers == mergedBinding.pImmutableSamplers, "{} != {} (binding #{}: {}, {})", fmt::ptr(binding.binding.pImmutableSamplers), fmt::ptr(mergedBinding.pImmutableSamplers), b, n, t);
                    mergedBinding.stageFlags |= binding.binding.stageFlags;
                    isMultistage = true;
                    break;
                }
                ++b;
            }
            if (!isMultistage) {
                size_t index = std::size(mergedBindings.bindings);
                mergedBindings.bindings.push_back(binding.binding);
                if (!mergedBindings.bindingIndices.emplace(bindingName, index).second) {
                    INVARIANT(false, "");
                }
                mergedBindings.bindingNames.push_back(std::move(bindingName));
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

    if (!std::empty(shaderModuleReflection.specializationConstants)) {
        if (!specializationConstants.emplace(shaderModule.getStage(), shaderModuleReflection.specializationConstants).second) {
            INVARIANT(false, "");
        }
    }
}

void ShaderStages::createDescriptorSetLayouts(std::string_view name, vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags)
{
    size_t setCount = std::size(setBindings);
    descriptorSetLayoutCreateInfoChains.reserve(setCount);
    descriptorSetLayoutHolders.reserve(setCount);
    descriptorSetLayouts.reserve(setCount);

    const auto & device = context.getDevice();

    descriptorSetLayoutCreateInfoChains.reserve(std::size(setBindings));
    for (const auto & [set, descriptorSetLayoutBindings] : setBindings) {
        auto & descriptorSetLayoutCreateInfoChain = descriptorSetLayoutCreateInfoChains.emplace_back();
        auto & descriptorSetLayoutCreateInfo = descriptorSetLayoutCreateInfoChain.get<vk::DescriptorSetLayoutCreateInfo>();
        descriptorSetLayoutCreateInfo.flags = descriptorSetLayoutCreateFlags;
        descriptorSetLayoutCreateInfo.setBindings(descriptorSetLayoutBindings.bindings);
        auto & descriptorSetLayoutBindingFlagsCreateInfo = descriptorSetLayoutCreateInfoChain.get<vk::DescriptorSetLayoutBindingFlagsCreateInfo>();
        descriptorSetLayoutBindingFlagsCreateInfo.setBindingFlags(nullptr);  // TODO:
        descriptorSetLayoutHolders.push_back(device.getHandle().createDescriptorSetLayoutUnique(descriptorSetLayoutCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
        descriptorSetLayouts.push_back(*descriptorSetLayoutHolders.back());

        for (const auto & descriptorSetLayoutBinding : descriptorSetLayoutBindings.bindings) {
            SPDLOG_DEBUG("BINDING ({}): set={} binding={} type={} stages={}", name, set, descriptorSetLayoutBinding.binding, descriptorSetLayoutBinding.descriptorType, descriptorSetLayoutBinding.stageFlags);
            if (descriptorSetLayoutCreateFlags & vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT) {
                INVARIANT(descriptorSetLayoutBinding.descriptorType != vk::DescriptorType::eUniformBufferDynamic, "Not compatible with eDescriptorBufferEXT descriptor set layout");
                INVARIANT(descriptorSetLayoutBinding.descriptorType != vk::DescriptorType::eStorageBufferDynamic, "Not compatible with eDescriptorBufferEXT descriptor set layout");
            } else {
                setDescriptorCounts[set][descriptorSetLayoutBinding.descriptorType] += descriptorSetLayoutBinding.descriptorCount;
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

    pushConstantRanges = mergePushConstantRanges(pushConstantRanges);
}

size_t ShaderStages::findSetByBindingName(const DescriptorBindingNameAndType & nameAndType) const
{
    for (const auto & [set, setBindings] : setBindings) {
        if (setBindings.bindingIndices.contains(nameAndType)) {
            return set;
        }
    }
    INVARIANT(false, "{}", nameAndType);
}

}  // namespace engine
