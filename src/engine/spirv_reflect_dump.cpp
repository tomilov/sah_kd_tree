#include <codegen/spirv_format.hpp>
#include <engine/spirv_reflect_dump.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <../SPIRV-Reflect/common/output_stream.h>
#include <../SPIRV-Reflect/spirv_reflect.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

#include <cstddef>
#include <cstdint>

namespace engine
{

static constexpr bool kUseSpirvReflectConversion = false;

namespace
{

[[nodiscard]] const char * toString(SpvReflectGenerator generator)
{
    switch (generator) {
    case SPV_REFLECT_GENERATOR_KHRONOS_LLVM_SPIRV_TRANSLATOR:
        return "KHRONOS_LLVM_SPIRV_TRANSLATOR";
    case SPV_REFLECT_GENERATOR_KHRONOS_SPIRV_TOOLS_ASSEMBLER:
        return "KHRONOS_SPIRV_TOOLS_ASSEMBLER";
    case SPV_REFLECT_GENERATOR_KHRONOS_GLSLANG_REFERENCE_FRONT_END:
        return "KHRONOS_GLSLANG_REFERENCE_FRONT_END";
    case SPV_REFLECT_GENERATOR_GOOGLE_SHADERC_OVER_GLSLANG:
        return "GOOGLE_SHADERC_OVER_GLSLANG";
    case SPV_REFLECT_GENERATOR_GOOGLE_SPIREGG:
        return "GOOGLE_SPIREGG";
    case SPV_REFLECT_GENERATOR_GOOGLE_RSPIRV:
        return "GOOGLE_RSPIRV";
    case SPV_REFLECT_GENERATOR_X_LEGEND_MESA_MESAIR_SPIRV_TRANSLATOR:
        return "X_LEGEND_MESA_MESAIR_SPIRV_TRANSLATOR";
    case SPV_REFLECT_GENERATOR_KHRONOS_SPIRV_TOOLS_LINKER:
        return "KHRONOS_SPIRV_TOOLS_LINKER";
    case SPV_REFLECT_GENERATOR_WINE_VKD3D_SHADER_COMPILER:
        return "WINE_VKD3D_SHADER_COMPILER";
    case SPV_REFLECT_GENERATOR_CLAY_CLAY_SHADER_COMPILER:
        return "CLAY_CLAY_SHADER_COMPILER";
    }
    INVARIANT(false, "Unknown SpvReflectGenerator value {}", fmt::underlying(generator));
}

[[nodiscard]] const char * toString(SpvReflectShaderStageFlagBits shaderStage)
{
    switch (shaderStage) {
    case SPV_REFLECT_SHADER_STAGE_VERTEX_BIT:
        return "VERTEX_BIT";
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        return "TESSELLATION_CONTROL_BIT";
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        return "TESSELLATION_EVALUATION_BIT";
    case SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT:
        return "GEOMETRY_BIT";
    case SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT:
        return "FRAGMENT_BIT";
    case SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT:
        return "COMPUTE_BIT";
    case SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV:
        return "TASK_BIT_NV";
    case SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV:
        return "MESH_BIT_NV";
    case SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR:
        return "RAYGEN_BIT_KHR";
    case SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR:
        return "ANY_HIT_BIT_KHR";
    case SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
        return "CLOSEST_HIT_BIT_KHR";
    case SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR:
        return "MISS_BIT_KHR";
    case SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR:
        return "INTERSECTION_BIT_KHR";
    case SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR:
        return "CALLABLE_BIT_KHR";
    }
    INVARIANT(false, "Unknown SpvReflectShaderStageFlagBits value {}", fmt::underlying(shaderStage));
}

[[nodiscard]] const char * toString(SpvReflectResourceType resourceType)
{
    switch (resourceType) {
    case SPV_REFLECT_RESOURCE_FLAG_UNDEFINED:
        return "UNDEFINED";
    case SPV_REFLECT_RESOURCE_FLAG_SAMPLER:
        return "SAMPLER";
    case SPV_REFLECT_RESOURCE_FLAG_CBV:
        return "CBV";
    case SPV_REFLECT_RESOURCE_FLAG_SRV:
        return "SRV";
    case SPV_REFLECT_RESOURCE_FLAG_UAV:
        return "UAV";
    }
    INVARIANT(false, "Unknown SpvReflectResourceType value {}", fmt::underlying(resourceType));
}

[[nodiscard]] const char * toString(SpvReflectDescriptorType descriptorType)
{
    switch (descriptorType) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        return "SAMPLER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return "COMBINED_IMAGE_SAMPLER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return "SAMPLED_IMAGE";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return "STORAGE_IMAGE";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return "UNIFORM_TEXEL_BUFFER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return "STORAGE_TEXEL_BUFFER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return "UNIFORM_BUFFER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return "STORAGE_BUFFER";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return "UNIFORM_BUFFER_DYNAMIC";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return "STORAGE_BUFFER_DYNAMIC";
    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return "INPUT_ATTACHMENT";
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return "ACCELERATION_STRUCTURE_KHR";
    }
    INVARIANT(false, "Unknown SpvReflectDescriptorType value {}", fmt::underlying(descriptorType));
}

[[nodiscard]] const char * toString(SpvReflectFormat reflectFormat)
{
    switch (reflectFormat) {
    case SPV_REFLECT_FORMAT_UNDEFINED:
        return "UNDEFINED";
    case SPV_REFLECT_FORMAT_R16_UINT:
        return "R16_UINT";
    case SPV_REFLECT_FORMAT_R16_SINT:
        return "R16_SINT";
    case SPV_REFLECT_FORMAT_R16_SFLOAT:
        return "R16_SFLOAT";
    case SPV_REFLECT_FORMAT_R16G16_UINT:
        return "R16G16_UINT";
    case SPV_REFLECT_FORMAT_R16G16_SINT:
        return "R16G16_SINT";
    case SPV_REFLECT_FORMAT_R16G16_SFLOAT:
        return "R16G16_SFLOAT";
    case SPV_REFLECT_FORMAT_R16G16B16_UINT:
        return "R16G16B16_UINT";
    case SPV_REFLECT_FORMAT_R16G16B16_SINT:
        return "R16G16B16_SINT";
    case SPV_REFLECT_FORMAT_R16G16B16_SFLOAT:
        return "R16G16B16_SFLOAT";
    case SPV_REFLECT_FORMAT_R16G16B16A16_UINT:
        return "R16G16B16A16_UINT";
    case SPV_REFLECT_FORMAT_R16G16B16A16_SINT:
        return "R16G16B16A16_SINT";
    case SPV_REFLECT_FORMAT_R16G16B16A16_SFLOAT:
        return "R16G16B16A16_SFLOAT";
    case SPV_REFLECT_FORMAT_R32_UINT:
        return "R32_UINT";
    case SPV_REFLECT_FORMAT_R32_SINT:
        return "R32_SINT";
    case SPV_REFLECT_FORMAT_R32_SFLOAT:
        return "R32_SFLOAT";
    case SPV_REFLECT_FORMAT_R32G32_UINT:
        return "R32G32_UINT";
    case SPV_REFLECT_FORMAT_R32G32_SINT:
        return "R32G32_SINT";
    case SPV_REFLECT_FORMAT_R32G32_SFLOAT:
        return "R32G32_SFLOAT";
    case SPV_REFLECT_FORMAT_R32G32B32_UINT:
        return "R32G32B32_UINT";
    case SPV_REFLECT_FORMAT_R32G32B32_SINT:
        return "R32G32B32_SINT";
    case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT:
        return "R32G32B32_SFLOAT";
    case SPV_REFLECT_FORMAT_R32G32B32A32_UINT:
        return "R32G32B32A32_UINT";
    case SPV_REFLECT_FORMAT_R32G32B32A32_SINT:
        return "R32G32B32A32_SINT";
    case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT:
        return "R32G32B32A32_SFLOAT";
    case SPV_REFLECT_FORMAT_R64_UINT:
        return "R64_UINT";
    case SPV_REFLECT_FORMAT_R64_SINT:
        return "R64_SINT";
    case SPV_REFLECT_FORMAT_R64_SFLOAT:
        return "R64_SFLOAT";
    case SPV_REFLECT_FORMAT_R64G64_UINT:
        return "R64G64_UINT";
    case SPV_REFLECT_FORMAT_R64G64_SINT:
        return "R64G64_SINT";
    case SPV_REFLECT_FORMAT_R64G64_SFLOAT:
        return "R64G64_SFLOAT";
    case SPV_REFLECT_FORMAT_R64G64B64_UINT:
        return "R64G64B64_UINT";
    case SPV_REFLECT_FORMAT_R64G64B64_SINT:
        return "R64G64B64_SINT";
    case SPV_REFLECT_FORMAT_R64G64B64_SFLOAT:
        return "R64G64B64_SFLOAT";
    case SPV_REFLECT_FORMAT_R64G64B64A64_UINT:
        return "R64G64B64A64_UINT";
    case SPV_REFLECT_FORMAT_R64G64B64A64_SINT:
        return "R64G64B64A64_SINT";
    case SPV_REFLECT_FORMAT_R64G64B64A64_SFLOAT:
        return "R64G64B64A64_SFLOAT";
    }
    INVARIANT(false, "Unknown SpvReflectFormat value {}", fmt::underlying(reflectFormat));
}

[[nodiscard]] const char * toString [[gnu::used]] (SpvReflectTypeFlagBits typeFlagBits)
{
    switch (typeFlagBits) {
    case SPV_REFLECT_TYPE_FLAG_UNDEFINED:
        return "UNDEFINED";
    case SPV_REFLECT_TYPE_FLAG_VOID:
        return "VOID";
    case SPV_REFLECT_TYPE_FLAG_BOOL:
        return "BOOL";
    case SPV_REFLECT_TYPE_FLAG_INT:
        return "INT";
    case SPV_REFLECT_TYPE_FLAG_FLOAT:
        return "FLOAT";
    case SPV_REFLECT_TYPE_FLAG_VECTOR:
        return "VECTOR";
    case SPV_REFLECT_TYPE_FLAG_MATRIX:
        return "MATRIX";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE:
        return "EXTERNAL_IMAGE";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLER:
        return "EXTERNAL_SAMPLER";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLED_IMAGE:
        return "EXTERNAL_SAMPLED_IMAGE";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_BLOCK:
        return "EXTERNAL_BLOCK";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE:
        return "EXTERNAL_ACCELERATION_STRUCTURE";
    case SPV_REFLECT_TYPE_FLAG_EXTERNAL_MASK:
        return "EXTERNAL_MASK";
    case SPV_REFLECT_TYPE_FLAG_STRUCT:
        return "STRUCT";
    case SPV_REFLECT_TYPE_FLAG_ARRAY:
        return "ARRAY";
    case SPV_REFLECT_TYPE_FLAG_REF:
        return "REF";
    }
    INVARIANT(false, "Unknown SpvReflectTypeFlagBits value {}", fmt::underlying(typeFlagBits));
}

[[nodiscard]] const char * toString [[gnu::used]] (SpvReflectVariableFlagBits ariableFlagBits)
{
    switch (ariableFlagBits) {
    case SPV_REFLECT_VARIABLE_FLAGS_NONE:
        return "NONE";
    case SPV_REFLECT_VARIABLE_FLAGS_UNUSED:
        return "UNUSED";
    case SPV_REFLECT_VARIABLE_FLAGS_PHYSICAL_POINTER_COPY:
        return "PHYSICAL_POINTER_COPY";
    }
    INVARIANT(false, "Unknown SpvReflectVariableFlagBits value {}", fmt::underlying(ariableFlagBits));
}

[[nodiscard]] const char * toString [[gnu::used]] (SpvReflectDecorationFlagBits decorationFlagBits)
{
    switch (decorationFlagBits) {
    case SPV_REFLECT_DECORATION_NONE:
        return "NONE";
    case SPV_REFLECT_DECORATION_BLOCK:
        return "BLOCK";
    case SPV_REFLECT_DECORATION_BUFFER_BLOCK:
        return "BUFFER_BLOCK";
    case SPV_REFLECT_DECORATION_ROW_MAJOR:
        return "ROW_MAJOR";
    case SPV_REFLECT_DECORATION_COLUMN_MAJOR:
        return "COLUMN_MAJOR";
    case SPV_REFLECT_DECORATION_BUILT_IN:
        return "BUILT_IN";
    case SPV_REFLECT_DECORATION_NOPERSPECTIVE:
        return "NOPERSPECTIVE";
    case SPV_REFLECT_DECORATION_FLAT:
        return "FLAT";
    case SPV_REFLECT_DECORATION_NON_WRITABLE:
        return "NON_WRITABLE";
    case SPV_REFLECT_DECORATION_RELAXED_PRECISION:
        return "RELAXED_PRECISION";
    case SPV_REFLECT_DECORATION_NON_READABLE:
        return "NON_READABLE";
    case SPV_REFLECT_DECORATION_PATCH:
        return "PATCH";
    case SPV_REFLECT_DECORATION_PER_VERTEX:
        return "PER_VERTEX";
    case SPV_REFLECT_DECORATION_PER_TASK:
        return "PER_TASK";
    case SPV_REFLECT_DECORATION_WEIGHT_TEXTURE:
        return "WEIGHT_TEXTURE";
    case SPV_REFLECT_DECORATION_BLOCK_MATCH_TEXTURE:
        return "BLOCK_MATCH_TEXTURE";
    }
    INVARIANT(false, "Unknown SpvReflectDecorationFlagBits value {}", fmt::underlying(decorationFlagBits));
}

template<typename T>
const char * toStringSpv(T value)
{
    const char * str = codegen::spv::toString(value);
    return str ? str : "???";
}

template<typename FlagBits>
struct Flags
{
    uint32_t value;
};

template<typename T>
struct Nullable
{
    T * value;

    Nullable(T * value) : value{value}
    {}
};

template<typename T>
Nullable(T * value) -> Nullable<T>;

template<typename T>
struct List
{
    T * p;
    uint32_t count;

    List(T * p, uint32_t count) : p{p}, count{count}
    {}
};

template<typename T>
List(T * p, uint32_t count) -> List<T>;

struct ReflectionStreamedFmt
{
    const spv_reflect::ShaderModule & shaderModule;

    friend std::ostream & operator<< [[gnu::used]] (std::ostream & out, const ReflectionStreamedFmt & reflectionStreamedFmt)
    {
        WriteReflection(reflectionStreamedFmt.shaderModule, false, out);
        return out;
    }
};

struct JsonStreamedFmt
{
    const nlohmann::json & j;

    friend std::ostream & operator<< [[gnu::used]] (std::ostream & out, const JsonStreamedFmt & j)
    {
        return out << std::setw(2) << j.j;
    }
};

}  // namespace

}  // namespace engine

template<>
struct fmt::formatter<engine::ReflectionStreamedFmt> : fmt::ostream_formatter
{
};

template<>
struct fmt::formatter<engine::JsonStreamedFmt> : fmt::ostream_formatter
{
};

template<>
struct nlohmann::adl_serializer<SpvReflectGenerator>
{
    static void to_json(json & j, const SpvReflectGenerator & generator)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringGenerator(generator);
        } else {
            j = engine::toString(generator);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectShaderStageFlagBits>
{
    static void to_json(json & j, const SpvReflectShaderStageFlagBits & shaderStageFlagBits)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringShaderStage(shaderStageFlagBits);
        } else {
            j = engine::toString(shaderStageFlagBits);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectResourceType>
{
    static void to_json(json & j, const SpvReflectResourceType & resourceType)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringResourceType(resourceType);
        } else {
            j = engine::toString(resourceType);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectDescriptorType>
{
    static void to_json(json & j, const SpvReflectDescriptorType & descriptorType)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringDescriptorType(descriptorType);
        } else {
            j = engine::toString(descriptorType);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvDim>
{
    static void to_json(json & j, const SpvDim & dim)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringSpvDim(dim);
        } else {
            j = engine::toStringSpv(dim);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvImageFormat>
{
    static void to_json(json & j, const SpvImageFormat & imageFormat)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringSpvImageFormat(imageFormat);
        } else {
            j = engine::toStringSpv(imageFormat);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvExecutionModel>
{
    static void to_json(json & j, const SpvExecutionModel & executionModel)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringSpvExecutionModel(executionModel);
        } else {
            j = engine::toStringSpv(executionModel);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvExecutionMode>
{
    static void to_json(json & j, const SpvExecutionMode & executionMode)
    {
        j = engine::toStringSpv(executionMode);
    }
};

template<>
struct nlohmann::adl_serializer<SpvCapability>
{
    static void to_json(json & j, const SpvCapability & capability)
    {
        j = engine::toStringSpv(capability);
    }
};

template<>
struct nlohmann::adl_serializer<SpvStorageClass>
{
    static void to_json(json & j, const SpvStorageClass & storageClass)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringSpvStorageClass(storageClass);
        } else {
            j = engine::toStringSpv(storageClass);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvSourceLanguage>
{
    static void to_json(json & j, const SpvSourceLanguage & sourceLanguage)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringSpvSourceLanguage(sourceLanguage);
        } else {
            j = engine::toStringSpv(sourceLanguage);
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvOp>
{
    static void to_json(json & j, const SpvOp & spvOp)
    {
        j = engine::toStringSpv(spvOp);
    }
};

template<>
struct nlohmann::adl_serializer<SpvBuiltIn>
{
    static void to_json(json & j, const SpvBuiltIn & builtIn)
    {
        j = engine::toStringSpv(builtIn);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectFormat>
{
    static void to_json(json & j, const SpvReflectFormat & format)
    {
        if (engine::kUseSpirvReflectConversion) {
            j = ToStringFormat(format);
        } else {
            j = engine::toString(format);
        }
    }
};

template<typename FlagBits>
struct nlohmann::adl_serializer<engine::Flags<FlagBits>>
{
    static void to_json(json & j, const engine::Flags<FlagBits> & flags)
    {
        j = json::array();
        uint32_t mask = flags.value;
        while (mask != 0) {
            uint32_t nextMask = mask & (mask - 1);
            uint32_t bit = nextMask ^ mask;
            auto s = engine::toString(FlagBits(utils::autoCast(bit)));
            j.push_back(s);
            mask = nextMask;
        }
    }
};

template<typename T>
struct nlohmann::adl_serializer<engine::Nullable<T>>
{
    static void to_json(json & j, const engine::Nullable<T> & p)
    {
        if (!p.value) {
            return;
        }
        j = *p.value;
    }
};

template<typename T>
struct nlohmann::adl_serializer<engine::List<T>>
{
    static void to_json(json & j, const engine::List<T> & list)
    {
        if (!list.p) {
            return;
        }
        j = json::array();
        for (size_t i = 0; i < list.count; ++i) {
            if constexpr (std::is_pointer_v<T>) {
                if (list.p[i]) {
                    j.push_back(*list.p[i]);
                } else {
                    j.push_back(nullptr);
                }
            } else {
                j.push_back(list.p[i]);
            }
        }
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectImageTraits>
{
    static void to_json(json & j, const SpvReflectImageTraits & imageTraits)
    {
        j.emplace("dim", imageTraits.dim);
        j.emplace("depth", imageTraits.depth);
        j.emplace("arrayed", imageTraits.arrayed);
        j.emplace("ms", imageTraits.ms);
        j.emplace("sampled", imageTraits.sampled);
        j.emplace("image_format", imageTraits.image_format);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectNumericTraits>
{
    static void to_json(json & j, const SpvReflectNumericTraits & numericTraits)
    {
        j.emplace("scalar", json::object({{"width", numericTraits.scalar.width}, {"signedness", numericTraits.scalar.signedness}}));
        j.emplace("vector", json::object({{"component_count", numericTraits.vector.component_count}}));
        j.emplace("matrix", json::object({{"column_count", numericTraits.matrix.column_count}, {"row_count", numericTraits.matrix.row_count}, {"stride", numericTraits.matrix.stride}}));
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectArrayTraits>
{
    static void to_json(json & j, const SpvReflectArrayTraits & arrayTraits)
    {
        j.emplace("dims_count", arrayTraits.dims_count);
        j.emplace("dims", engine::List(arrayTraits.dims, arrayTraits.dims_count));
        j.emplace("spec_constant_op_ids", engine::List(arrayTraits.spec_constant_op_ids, arrayTraits.dims_count));
        j.emplace("stride", arrayTraits.stride);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectTypeDescription>
{
    static void to_json(json & j, const SpvReflectTypeDescription & typeDescription)
    {
        j.emplace("id", typeDescription.id);
        j.emplace("op", typeDescription.op);
        j.emplace("type_name", typeDescription.type_name ? typeDescription.type_name : "");
        j.emplace("struct_member_name", typeDescription.struct_member_name ? typeDescription.struct_member_name : "");
        j.emplace("storage_class", typeDescription.storage_class);
        j.emplace("type_flags", engine::Flags<SpvReflectTypeFlagBits>{typeDescription.type_flags});
        j.emplace("decoration_flags", engine::Flags<SpvReflectDecorationFlagBits>{typeDescription.decoration_flags});
        j.emplace("traits", json::object({{"numeric", typeDescription.traits.numeric}, {"image", typeDescription.traits.image}, {"array", typeDescription.traits.array}}));
        j.emplace("member_count", typeDescription.member_count);
        j.emplace("members", engine::List(typeDescription.members, typeDescription.member_count));
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectBlockVariable>
{
    static void to_json(json & j, const SpvReflectBlockVariable & reflectBlockVariable)
    {
        j.emplace("spirv_id", reflectBlockVariable.spirv_id);
        j.emplace("name", reflectBlockVariable.name ? reflectBlockVariable.name : "");
        j.emplace("offset", reflectBlockVariable.offset);
        j.emplace("absolute_offset", reflectBlockVariable.absolute_offset);
        j.emplace("size", reflectBlockVariable.size);
        j.emplace("padded_size", reflectBlockVariable.padded_size);
        j.emplace("decoration_flags", engine::Flags<SpvReflectDecorationFlagBits>{reflectBlockVariable.decoration_flags});
        j.emplace("numeric", reflectBlockVariable.numeric);
        j.emplace("array", reflectBlockVariable.array);
        j.emplace("flags", engine::Flags<SpvReflectVariableFlagBits>{reflectBlockVariable.flags});
        j.emplace("member_count", reflectBlockVariable.member_count);
        j.emplace("members", engine::List(reflectBlockVariable.members, reflectBlockVariable.member_count));
        j.emplace("type_description", engine::Nullable(reflectBlockVariable.type_description));
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectBindingArrayTraits>
{
    static void to_json(json & j, const SpvReflectBindingArrayTraits & bindingArrayTraits)
    {
        j.emplace("dims_count", bindingArrayTraits.dims_count);
        j.emplace("dims", engine::List(bindingArrayTraits.dims, bindingArrayTraits.dims_count));
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectDescriptorBinding>
{
    static void to_json(json & j, const SpvReflectDescriptorBinding & descriptorBinding)
    {
        j.emplace("spirv_id", descriptorBinding.spirv_id);
        j.emplace("name", descriptorBinding.name ? descriptorBinding.name : "");
        j.emplace("binding", descriptorBinding.binding);
        j.emplace("input_attachment_index", descriptorBinding.input_attachment_index);
        j.emplace("set", descriptorBinding.set);
        j.emplace("descriptor_type", descriptorBinding.descriptor_type);
        j.emplace("resource_type", descriptorBinding.resource_type);
        j.emplace("image", descriptorBinding.image);
        j.emplace("block", descriptorBinding.block);
        j.emplace("array", descriptorBinding.array);
        j.emplace("count", descriptorBinding.count);
        j.emplace("accessed", descriptorBinding.accessed);
        j.emplace("uav_counter_id", descriptorBinding.uav_counter_id);
        j.emplace("uav_counter_binding", engine::Nullable(descriptorBinding.uav_counter_binding));
        j.emplace("type_description", engine::Nullable(descriptorBinding.type_description));
        j.emplace("word_offset", json::object({{"binding", descriptorBinding.word_offset.binding}, {"set", descriptorBinding.word_offset.set}}));
        j.emplace("decoration_flags", engine::Flags<SpvReflectDecorationFlagBits>{descriptorBinding.decoration_flags});
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectDescriptorSet>
{
    static void to_json(json & j, const SpvReflectDescriptorSet & descriptorSet)
    {
        j.emplace("set", descriptorSet.set);
        j.emplace("binding_count", descriptorSet.binding_count);
        j.emplace("bindings", engine::List(descriptorSet.bindings, descriptorSet.binding_count));
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectInterfaceVariable>
{
    static void to_json(json & j, const SpvReflectInterfaceVariable & interfaceVariable)
    {
        j.emplace("spirv_id", interfaceVariable.spirv_id);
        j.emplace("name", interfaceVariable.name ? interfaceVariable.name : "");
        j.emplace("location", interfaceVariable.location);
        j.emplace("storage_class", interfaceVariable.storage_class);
        j.emplace("semantic", interfaceVariable.semantic ? interfaceVariable.semantic : "");
        j.emplace("decoration_flags", engine::Flags<SpvReflectDecorationFlagBits>{interfaceVariable.decoration_flags});
        j.emplace("built_in", interfaceVariable.built_in);
        j.emplace("numeric", interfaceVariable.numeric);
        j.emplace("array", interfaceVariable.array);
        j.emplace("member_count", interfaceVariable.member_count);
        j.emplace("members", engine::List(interfaceVariable.members, interfaceVariable.member_count));
        j.emplace("format", interfaceVariable.format);
        j.emplace("type_description", engine::Nullable(interfaceVariable.type_description));
        j.emplace("location", interfaceVariable.word_offset.location);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectEntryPoint>
{
    static void to_json(json & j, const SpvReflectEntryPoint & entryPoint)
    {
        j.emplace("name", entryPoint.name ? entryPoint.name : "");
        j.emplace("id", entryPoint.id);
        j.emplace("spirv_execution_model", entryPoint.spirv_execution_model);
        j.emplace("shader_stage", entryPoint.shader_stage);
        j.emplace("input_variable_count", entryPoint.input_variable_count);
        j.emplace("input_variables", engine::List(entryPoint.input_variables, entryPoint.input_variable_count));
        j.emplace("output_variable_count", entryPoint.output_variable_count);
        j.emplace("output_variables", engine::List(entryPoint.output_variables, entryPoint.output_variable_count));
        j.emplace("interface_variable_count", entryPoint.interface_variable_count);
        j.emplace("interface_variables", engine::List(entryPoint.interface_variables, entryPoint.interface_variable_count));
        j.emplace("descriptor_set_count", entryPoint.descriptor_set_count);
        j.emplace("descriptor_sets", engine::List(entryPoint.descriptor_sets, entryPoint.descriptor_set_count));
        j.emplace("used_uniform_count", entryPoint.used_uniform_count);
        j.emplace("used_uniforms", engine::List(entryPoint.used_uniforms, entryPoint.used_uniform_count));
        j.emplace("used_push_constant_count", entryPoint.used_push_constant_count);
        j.emplace("used_push_constants", engine::List(entryPoint.used_push_constants, entryPoint.used_push_constant_count));
        j.emplace("execution_mode_count", entryPoint.execution_mode_count);
        j.emplace("execution_modes", engine::List(entryPoint.execution_modes, entryPoint.execution_mode_count));
        j.emplace("local_size", json::object({{"x", entryPoint.local_size.x}, {"y", entryPoint.local_size.y}, {"z", entryPoint.local_size.z}}));
        j.emplace("invocations", entryPoint.invocations);
        j.emplace("output_vertices", entryPoint.output_vertices);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectCapability>
{
    static void to_json(json & j, const SpvReflectCapability & capability)
    {
        j.emplace("value", capability.value);
        j.emplace("word_offset", capability.word_offset);
    }
};

template<>
struct nlohmann::adl_serializer<SpvReflectShaderModule>
{
    static void to_json(json & j, const SpvReflectShaderModule & shaderModule)
    {
        j.emplace("generator", shaderModule.generator);
        j.emplace("entry_point_name", shaderModule.entry_point_name ? shaderModule.entry_point_name : "");
        j.emplace("entry_point_id", shaderModule.entry_point_id);
        j.emplace("entry_point_count", shaderModule.entry_point_count);
        j.emplace("entry_points", engine::List(shaderModule.entry_points, shaderModule.entry_point_count));
        j.emplace("source_language", shaderModule.source_language);
        j.emplace("source_language_version", shaderModule.source_language_version);
        j.emplace("source_file", shaderModule.source_file ? shaderModule.source_file : "");
        j.emplace("source_source", shaderModule.source_source ? shaderModule.source_source : "");
        j.emplace("capability_count", shaderModule.capability_count);
        j.emplace("capabilities", engine::List(shaderModule.capabilities, shaderModule.capability_count));
        j.emplace("spirv_execution_model", shaderModule.spirv_execution_model);
        j.emplace("shader_stage", shaderModule.shader_stage);
        j.emplace("descriptor_binding_count", shaderModule.descriptor_binding_count);
        j.emplace("descriptor_bindings", engine::List(shaderModule.descriptor_bindings, shaderModule.descriptor_binding_count));
        j.emplace("descriptor_set_count", shaderModule.descriptor_set_count);
        j.emplace("descriptor_sets", engine::List(shaderModule.descriptor_sets, shaderModule.descriptor_set_count));
        j.emplace("input_variable_count", shaderModule.input_variable_count);
        j.emplace("input_variables", engine::List(shaderModule.input_variables, shaderModule.input_variable_count));
        j.emplace("output_variable_count", shaderModule.output_variable_count);
        j.emplace("output_variables", engine::List(shaderModule.output_variables, shaderModule.output_variable_count));
        j.emplace("interface_variable_count", shaderModule.interface_variable_count);
        j.emplace("interface_variables", engine::List(shaderModule.interface_variables, shaderModule.interface_variable_count));
        j.emplace("push_constant_block_count", shaderModule.push_constant_block_count);
        j.emplace("push_constant_blocks", engine::List(shaderModule.push_constant_blocks, shaderModule.push_constant_block_count));
    }
};

namespace engine
{

void dump(const spv_reflect::ShaderModule & shaderModule)
{
    if ((true)) {
        nlohmann::json j = shaderModule.GetShaderModule();
        SPDLOG_INFO("ShaderModule: {}", JsonStreamedFmt{j});
    }
    if ((false)) {
        SPDLOG_INFO("ShaderModule: {}", ReflectionStreamedFmt{shaderModule});
    }
    if ((false)) {
        SpvReflectToYaml spvReflectToYaml{shaderModule.GetShaderModule(), 0};
        // SPDLOG_INFO("ShaderModule: {}", fmt::streamed(spvReflectToYaml)); // sadly operator << expect non-const ref
        std::ostringstream oss;
        oss << spvReflectToYaml;
        SPDLOG_INFO("ShaderModule: {}", oss.str());
    }
}

std::string serialize(const SpvReflectDescriptorSet & descriptorSet)
{
    nlohmann::json j = descriptorSet;
    return fmt::to_string(JsonStreamedFmt{j});
}

}  // namespace engine
