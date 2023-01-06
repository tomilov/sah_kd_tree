#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/file_io.hpp>
#include <engine/format.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <codegen/spirv_format.hpp>

#include <../SPIRV-Reflect/common/output_stream.h>
#include <../SPIRV-Reflect/spirv_reflect.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <iterator>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cstddef>
#include <cstdint>

namespace engine
{

namespace
{

constexpr bool kUseSpirvReflectConversion = false;

[[nodiscard]] const char * toString(SpvReflectResult result)
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
    }
    INVARIANT(false, "Unknown SpvReflectResult value {}", fmt::underlying(result));
}

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

[[nodiscard]] const char * toString(SpvReflectTypeFlagBits typeFlagBits)
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
    }
    INVARIANT(false, "Unknown SpvReflectTypeFlagBits value {}", fmt::underlying(typeFlagBits));
}

[[nodiscard]] const char * toString(SpvReflectVariableFlagBits ariableFlagBits)
{
    switch (ariableFlagBits) {
    case SPV_REFLECT_VARIABLE_FLAGS_NONE:
        return "NONE";
    case SPV_REFLECT_VARIABLE_FLAGS_UNUSED:
        return "UNUSED";
    }
    INVARIANT(false, "Unknown SpvReflectVariableFlagBits value {}", fmt::underlying(ariableFlagBits));
}

[[nodiscard]] const char * toString(SpvReflectDecorationFlagBits decorationFlagBits)
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
    }
    INVARIANT(false, "Unknown SpvReflectDecorationFlagBits value {}", fmt::underlying(decorationFlagBits));
}

template<typename T>
auto toStringSpv(T value)
{
    auto str = codegen::spv::toString(value);
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
};

template<typename T>
constexpr Nullable<T> markAsNullable(T * value) noexcept
{
    return {value};
}

template<typename T>
struct PtrList
{
    T ** p;
    uint32_t count;
};

template<typename T>
constexpr PtrList<T> makePtrList(T ** p, uint32_t count) noexcept
{
    return {p, count};
}

template<typename T>
struct EnumList
{
    T * p;
    uint32_t count;
};

template<typename T>
constexpr auto makeList(T * p, uint32_t count) noexcept
{
    if constexpr (std::is_enum_v<T>) {
        return EnumList<T>{p, count};
    } else {
        return fmt::join(p, p + count, ",");
    }
}

struct SourceCodeJsonEscape
{
    const char * text;
};

std::string recurseFormat(const SpvReflectBlockVariable * members, uint32_t memberCount);
std::string recurseFormat(const SpvReflectTypeDescription * members, uint32_t memberCount);
std::string recurseFormat(const SpvReflectDescriptorBinding * nullable);
std::string recurseFormat(const SpvReflectInterfaceVariable * members, uint32_t memberCount);

struct ReflectionStreamedFmt
{
    const spv_reflect::ShaderModule & shaderModule;

    friend std::ostream & operator<< [[maybe_unused]] (std::ostream & out, const ReflectionStreamedFmt & reflectionStreamedFmt)
    {
        WriteReflection(reflectionStreamedFmt.shaderModule, false, out);
        return out;
    }
};

}  // namespace

}  // namespace engine

template<>
struct fmt::formatter<engine::ReflectionStreamedFmt> : fmt::ostream_formatter
{
};

template<>
struct fmt::formatter<SpvReflectResult> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectResult reflectResult, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(engine::toString(reflectResult), ctx);
    }
};

template<>
struct fmt::formatter<SpvReflectGenerator> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectGenerator generator, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringGenerator(generator), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toString(generator), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvReflectShaderStageFlagBits> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectShaderStageFlagBits shaderStageFlagBits, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringShaderStage(shaderStageFlagBits), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toString(shaderStageFlagBits), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvReflectResourceType> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectResourceType resourceType, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringResourceType(resourceType), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toString(resourceType), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvReflectDescriptorType> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectDescriptorType descriptorType, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringDescriptorType(descriptorType), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toString(descriptorType), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvDim> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvDim dim, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvDim(dim), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(dim), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvImageFormat> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvImageFormat imageFormat, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvImageFormat(imageFormat), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(imageFormat), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvExecutionModel> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvExecutionModel executionModel, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvExecutionModel(executionModel), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(executionModel), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvExecutionMode> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvExecutionMode executionMode, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(executionMode), ctx);
    }
};

template<>
struct fmt::formatter<SpvCapability> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvCapability capability, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(capability), ctx);
    }
};

template<>
struct fmt::formatter<SpvStorageClass> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvStorageClass storageClass, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvStorageClass(storageClass), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(storageClass), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvSourceLanguage> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvSourceLanguage sourceLanguage, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvSourceLanguage(sourceLanguage), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(sourceLanguage), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvOp> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvOp spvOp, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(spvOp), ctx);
    }
};

template<>
struct fmt::formatter<SpvBuiltIn> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvBuiltIn builtIn, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringSpvBuiltIn(builtIn), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toStringSpv(builtIn), ctx);
        }
    }
};

template<>
struct fmt::formatter<SpvReflectFormat> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(SpvReflectFormat format, FormatContext & ctx) const
    {
        if (engine::kUseSpirvReflectConversion) {
            return fmt::formatter<fmt::string_view>::format(ToStringFormat(format), ctx);
        } else {
            return fmt::formatter<fmt::string_view>::format(engine::toString(format), ctx);
        }
    }
};

template<typename FlagBits>
struct fmt::formatter<engine::Flags<FlagBits>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const engine::Flags<FlagBits> & flags, FormatContext & ctx) const
    {
        auto out = ctx.out();
        *out++ = '[';
        uint32_t mask = flags.value;
        while (mask != 0) {
            uint32_t nextMask = mask & (mask - 1);
            uint32_t bit = nextMask ^ mask;
            auto s = engine::toString(FlagBits(utils::autoCast(bit)));
            out = fmt::format_to(out, R"json("{}")json", s);
            if (nextMask != 0) {
                *out++ = ',';
            }
            mask = nextMask;
        }
        *out++ = ']';
        return out;
    }
};

template<typename T>
struct fmt::formatter<engine::Nullable<T>> : fmt::formatter<T>
{
    template<typename FormatContext>
    auto format(const engine::Nullable<T> & p, FormatContext & ctx) const
    {
        if (!p.value) {
            return fmt::format_to(ctx.out(), "null");
        }
        return fmt::formatter<T>::format(*p.value, ctx);
    }
};

template<typename T>
struct fmt::formatter<engine::PtrList<T>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const engine::PtrList<T> & ptrList, FormatContext & ctx) const
    {
        auto out = ctx.out();
        if (!ptrList.p) {
            return out;
        }
        if (ptrList.count > 0) {
            INVARIANT(ptrList.p[0], "");
            out = fmt::format_to(out, "{}", *ptrList.p[0]);
            for (size_t i = 1; i < ptrList.count; ++i) {
                const T * v = ptrList.p[i];
                INVARIANT(v, "");
                out = fmt::format_to(out, ",{}", *v);
            }
        }
        return out;
    }
};

template<typename T>
struct fmt::formatter<engine::EnumList<T>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const engine::EnumList<T> & enumList, FormatContext & ctx) const
    {
        auto out = ctx.out();
        if (!enumList.p) {
            return out;
        }
        if (enumList.count > 0) {
            out = fmt::format_to(out, R"json("{}")json", enumList.p[0]);
            for (size_t i = 1; i < enumList.count; ++i) {
                out = fmt::format_to(out, R"json(,"{}")json", enumList.p[i]);
            }
        }
        return out;
    }
};

template<>
struct fmt::formatter<engine::SourceCodeJsonEscape> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const engine::SourceCodeJsonEscape & sourceCodejsonEscape, FormatContext & ctx) const
    {
        auto out = ctx.out();
        std::string_view text = sourceCodejsonEscape.text;
        if (std::empty(text)) {
            return out;
        }
        using namespace std::string_view_literals;
        const auto append = [&out](std::string_view s) { out = std::copy_n(std::cbegin(s), std::size(s), out); };
        while (!std::empty(text)) {
            char c = text.front();
            switch (c) {
            case '\\':
                append("\\\\"sv);
                break;
            case '"':
                append("\\\""sv);
                break;
            case '\b':
                append("\\b"sv);
                break;
            case '\f':
                append("\\f"sv);
                break;
            case '\n':
                append("\\n"sv);
                break;
            case '\r':
                append("\\r"sv);
                break;
            case '\t':
                append("\\t"sv);
                break;
            default: {
                if (static_cast<uint8_t>(c) <= 0x1f) {
                    append("\\u"sv);
                    char buf[4] = {'0', '0', '0', '0'};
                    auto r = std::to_chars(std::begin(buf), std::end(buf), static_cast<uint8_t>(c), 16);
                    INVARIANT(r.ec == std::errc{}, "Failed to convert char('{}') to hexadecimal string of length 4", c);
                    std::rotate(std::begin(buf), r.ptr, std::end(buf));
                    append({std::data(buf), std::size(buf)});
                } else if (text.starts_with("\xE2\x80\xA8"sv)) {
                    append("\\u2028"sv);
                    text.remove_prefix(3);
                    continue;
                } else if (text.starts_with("\xE2\x80\xA9"sv)) {
                    append("\\u2029"sv);
                    text.remove_prefix(3);
                    continue;
                } else {
                    *out++ = c;
                }
            }
            }
            text.remove_prefix(1);
        }
        return out;
    }
};

template<>
struct fmt::formatter<SpvReflectImageTraits> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectImageTraits & imageTraits, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"dim":"{}","depth":{},"arrayed":{},"ms":{},"sampled":{},"image_format":"{}"}})json", imageTraits.dim, imageTraits.depth, imageTraits.arrayed, imageTraits.ms, imageTraits.sampled,
                              imageTraits.image_format);
    }
};

template<>
struct fmt::formatter<SpvReflectNumericTraits> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectNumericTraits & numericTraits, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"scalar":{{"width":{},"signedness":{}}},"vector":{{"component_count":{}}},"matrix":{{"column_count":{},"row_count":{},"stride":{}}}}})json", numericTraits.scalar.width,
                              numericTraits.scalar.signedness, numericTraits.vector.component_count, numericTraits.matrix.column_count, numericTraits.matrix.row_count, numericTraits.matrix.stride);
    }
};

template<>
struct fmt::formatter<SpvReflectArrayTraits> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectArrayTraits & arrayTraits, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"dims_count":{},"dims":[{}],"spec_constant_op_ids":[{}],"stride":{}}})json", arrayTraits.dims_count, engine::makeList(arrayTraits.dims, arrayTraits.dims_count),
                              engine::makeList(arrayTraits.spec_constant_op_ids, arrayTraits.dims_count), arrayTraits.stride);
    }
};

template<>
struct fmt::formatter<SpvReflectTypeDescription> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectTypeDescription & typeDescription, FormatContext & ctx) const
    {
        auto memberCount = typeDescription.member_count;
        return fmt::format_to(ctx.out(),
                              R"json({{"id":{},"op":"{}","type_name":"{}","struct_member_name":"{}","storage_class":"{}","type_flags":{},"decoration_flags":{},"traits":{{"numeric":{},"image":{},"array":{}}},"member_count":{},"members":[{}]}})json",
                              typeDescription.id, typeDescription.op, typeDescription.type_name ? typeDescription.type_name : "", typeDescription.struct_member_name ? typeDescription.struct_member_name : "", typeDescription.storage_class,
                              engine::Flags<SpvReflectTypeFlagBits>{typeDescription.type_flags}, engine::Flags<SpvReflectDecorationFlagBits>{typeDescription.decoration_flags}, typeDescription.traits.numeric, typeDescription.traits.image,
                              typeDescription.traits.array, memberCount, engine::recurseFormat(typeDescription.members, memberCount));
    }
};

template<>
struct fmt::formatter<SpvReflectBlockVariable> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectBlockVariable & reflectBlockVariable, FormatContext & ctx) const
    {
        auto memberCount = reflectBlockVariable.member_count;
        return fmt::format_to(ctx.out(),
                              R"json({{"spirv_id":{},"name":"{}","offset":{},"absolute_offset":{},"size":{},"padded_size":{},"decoration_flags":{},"numeric":{},"array":{},"flags":{},"member_count":{},"members":[{}],"type_description":{}}})json",
                              reflectBlockVariable.spirv_id, reflectBlockVariable.name ? reflectBlockVariable.name : "", reflectBlockVariable.offset, reflectBlockVariable.absolute_offset, reflectBlockVariable.size, reflectBlockVariable.padded_size,
                              engine::Flags<SpvReflectDecorationFlagBits>{reflectBlockVariable.decoration_flags}, reflectBlockVariable.numeric, reflectBlockVariable.array, engine::Flags<SpvReflectVariableFlagBits>{reflectBlockVariable.flags},
                              memberCount, engine::recurseFormat(reflectBlockVariable.members, memberCount), engine::markAsNullable(reflectBlockVariable.type_description));
    }
};

template<>
struct fmt::formatter<SpvReflectBindingArrayTraits> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectBindingArrayTraits & bindingArrayTraits, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"dims_count":{},"dims":[{}]}})json", bindingArrayTraits.dims_count, engine::makeList(bindingArrayTraits.dims, bindingArrayTraits.dims_count));
    }
};

template<>
struct fmt::formatter<SpvReflectDescriptorBinding> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectDescriptorBinding & descriptorBinding, FormatContext & ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            R"json({{"spirv_id":{},"name":"{}","binding":{},"input_attachment_index":{},"set":{},"descriptor_type":"{}","resource_type":"{}","image":{},"block":{},"array":{},"count":{},"accessed":{},"uav_counter_id":{},"uav_counter_binding":{},"type_description":{},"word_offset.binding":{},"word_offset.set":{},"decoration_flags":{}}})json",
            descriptorBinding.spirv_id, descriptorBinding.name ? descriptorBinding.name : "", descriptorBinding.binding, descriptorBinding.input_attachment_index, descriptorBinding.set, descriptorBinding.descriptor_type,
            descriptorBinding.resource_type, descriptorBinding.image, descriptorBinding.block, descriptorBinding.array, descriptorBinding.count, descriptorBinding.accessed, descriptorBinding.uav_counter_id,
            engine::recurseFormat(descriptorBinding.uav_counter_binding), engine::markAsNullable(descriptorBinding.type_description), descriptorBinding.word_offset.binding, descriptorBinding.word_offset.set,
            engine::Flags<SpvReflectDecorationFlagBits>{descriptorBinding.decoration_flags});
    }
};

template<>
struct fmt::formatter<SpvReflectDescriptorSet> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectDescriptorSet & descriptorSet, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"set":{},"binding_count":{},"bindings":[{}]}})json", descriptorSet.set, descriptorSet.binding_count, engine::makePtrList(descriptorSet.bindings, descriptorSet.binding_count));
    }
};

template<>
struct fmt::formatter<SpvReflectInterfaceVariable> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectInterfaceVariable & interfaceVariable, FormatContext & ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            R"json({{"spirv_id":{},"name":"{}","location":{},"storage_class":"{}","semantic":"{}","decoration_flags":{},"built_in":"{}","numeric":{},"array":{},"member_count":{},"members":[{}],"format":"{}","type_description":{},"word_offset":{{"location":{}}}}})json",
            interfaceVariable.spirv_id, interfaceVariable.name ? interfaceVariable.name : "", interfaceVariable.location, interfaceVariable.storage_class, interfaceVariable.semantic ? interfaceVariable.semantic : "",
            engine::Flags<SpvReflectDecorationFlagBits>{interfaceVariable.decoration_flags}, interfaceVariable.built_in, interfaceVariable.numeric, interfaceVariable.array, interfaceVariable.member_count,
            engine::recurseFormat(interfaceVariable.members, interfaceVariable.member_count), interfaceVariable.format, engine::markAsNullable(interfaceVariable.type_description), interfaceVariable.word_offset.location);
    }
};

template<>
struct fmt::formatter<SpvReflectEntryPoint> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectEntryPoint & entryPoint, FormatContext & ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            R"json({{"name":"{}","id":{},"spirv_execution_model":"{}","shader_stage":"{}","input_variable_count":{},"input_variables":[{}],"output_variable_count":{},"output_variables":[{}],"interface_variable_count":{},"interface_variables":[{}],"descriptor_set_count":{},"descriptor_sets":[{}],"used_uniform_count":{},"used_uniforms":[{}],"used_push_constant_count":{},"used_push_constants":[{}],"execution_mode_count":{},"execution_modes":[{}],"local_size":{{"x":{},"y":{},"z":{}}},"invocations":{},"output_vertices":{}}})json",
            entryPoint.name ? entryPoint.name : "", entryPoint.id, entryPoint.spirv_execution_model, entryPoint.shader_stage, entryPoint.input_variable_count, engine::makePtrList(entryPoint.input_variables, entryPoint.input_variable_count),
            entryPoint.output_variable_count, engine::makePtrList(entryPoint.output_variables, entryPoint.output_variable_count), entryPoint.interface_variable_count,
            engine::makeList(entryPoint.interface_variables, entryPoint.interface_variable_count), entryPoint.descriptor_set_count, engine::makeList(entryPoint.descriptor_sets, entryPoint.descriptor_set_count), entryPoint.used_uniform_count,
            engine::makeList(entryPoint.used_uniforms, entryPoint.used_uniform_count), entryPoint.used_push_constant_count, engine::makeList(entryPoint.used_push_constants, entryPoint.used_push_constant_count), entryPoint.execution_mode_count,
            engine::makeList(entryPoint.execution_modes, entryPoint.execution_mode_count), entryPoint.local_size.x, entryPoint.local_size.y, entryPoint.local_size.z, entryPoint.invocations, entryPoint.output_vertices);
    }
};

template<>
struct fmt::formatter<SpvReflectCapability> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectCapability & capability, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), R"json({{"value":"{}","word_offset":{}}})json", capability.value, capability.word_offset);
    }
};

template<>
struct fmt::formatter<SpvReflectShaderModule> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const SpvReflectShaderModule & shaderModule, FormatContext & ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            R"json({{"generator":"{}","entry_point_name":"{}","entry_point_id":{},"entry_point_count":{},"entry_points":[{}],"source_language":"{}","source_language_version":{},"source_file":"{}","source_source":"{}","capability_count":{},"capabilities":[{}],"spirv_execution_model":"{}","shader_stage":"{}","descriptor_binding_count":{},"descriptor_bindings":[{}],"descriptor_set_count":{},"descriptor_sets":[{}],"input_variable_count":{},"input_variables":[{}],"output_variable_count":{},"output_variables":[{}],"interface_variable_count":{},"interface_variables":[{}],"push_constant_block_count":{},"push_constant_blocks":[{}]}})json",
            shaderModule.generator, shaderModule.entry_point_name ? shaderModule.entry_point_name : "", shaderModule.entry_point_id, shaderModule.entry_point_count, engine::makeList(shaderModule.entry_points, shaderModule.entry_point_count),
            shaderModule.source_language, shaderModule.source_language_version, shaderModule.source_file ? shaderModule.source_file : "", engine::SourceCodeJsonEscape{shaderModule.source_source ? shaderModule.source_source : ""},
            shaderModule.capability_count, engine::makeList(shaderModule.capabilities, shaderModule.capability_count), shaderModule.spirv_execution_model, shaderModule.shader_stage, shaderModule.descriptor_binding_count,
            engine::makeList(shaderModule.descriptor_bindings, shaderModule.descriptor_binding_count), shaderModule.descriptor_set_count, engine::makeList(shaderModule.descriptor_sets, shaderModule.descriptor_set_count),
            shaderModule.input_variable_count, engine::makePtrList(shaderModule.input_variables, shaderModule.input_variable_count), shaderModule.output_variable_count,
            engine::makePtrList(shaderModule.output_variables, shaderModule.output_variable_count), shaderModule.interface_variable_count, engine::makeList(shaderModule.interface_variables, shaderModule.interface_variable_count),
            shaderModule.push_constant_block_count, engine::makeList(shaderModule.push_constant_blocks, shaderModule.push_constant_block_count));
    }
};

namespace engine
{

namespace
{

std::string recurseFormat(const SpvReflectBlockVariable * members, uint32_t memberCount)
{
    return fmt::to_string(makeList(members, memberCount));
}

std::string recurseFormat(const SpvReflectTypeDescription * members, uint32_t memberCount)
{
    return fmt::to_string(makeList(members, memberCount));
}

std::string recurseFormat(const SpvReflectDescriptorBinding * nullable)
{
    if (!nullable) {
        return "null";
    }
    return fmt::to_string(*nullable);
}

std::string recurseFormat(const SpvReflectInterfaceVariable * members, uint32_t memberCount)
{
    return fmt::to_string(makeList(members, memberCount));
}

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
    case vk::ShaderStageFlagBits::eSubpassShadingHUAWEI: {
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

ShaderModule::ShaderModule(std::string_view name, const Engine & engine, utils::CheckedPtr<const FileIo> fileIo) : name{name}, engine{engine}, fileIo{fileIo}, library{*engine.library}, device{*engine.device}
{
    load();
}

void ShaderModule::load()
{
    shaderStage = shaderNameToStage(name);
    INVARIANT(fileIo, "Exptected non-null");
    spirv = fileIo->loadShader(name);

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo.setCode(spirv);
    shaderModuleHolder = device.device.createShaderModuleUnique(shaderModuleCreateInfo, library.allocationCallbacks, library.dispatcher);
    shaderModule = *shaderModuleHolder;

    device.setDebugUtilsObjectName(shaderModule, name);
}

ShaderModuleReflection::ShaderModuleReflection(const ShaderModule & shaderModule) : shaderModule{shaderModule}, reflectionModule{shaderModule.spirv, SPV_REFLECT_MODULE_FLAG_NO_COPY}
{
    auto reflectionResult = reflectionModule->GetResult();
    INVARIANT(reflectionResult == SPV_REFLECT_RESULT_SUCCESS, "spvReflectCreateShaderModule returned {} for shader module '{}'", reflectionResult, shaderModule.name);

    reflect();
}

ShaderModuleReflection::~ShaderModuleReflection() = default;

void ShaderModuleReflection::reflect()
{
    SpvReflectResult reflectResult = SPV_REFLECT_RESULT_SUCCESS;

    shaderStage = spvReflectShaderStageToVk(reflectionModule->GetShaderStage());
    if (!(shaderStage & shaderModule.shaderStage)) {
        SPDLOG_WARN("Reflected flags ({}) of shader module '{}' does not contain inferred flags ({})", shaderStage, shaderModule.name, shaderModule.shaderStage);
    }

    if ((true)) {
        SPDLOG_INFO("ShaderModule: {}", reflectionModule->GetShaderModule());
    }
    if ((true)) {
        SPDLOG_INFO("ShaderModule: {}", ReflectionStreamedFmt{*reflectionModule});
    }
    if ((false)) {
        SpvReflectToYaml spvReflectToYaml{reflectionModule->GetShaderModule(), 0};
        // SPDLOG_INFO("ShaderModule: {}", fmt::streamed(spvReflectToYaml)); // sadly operator << expect non-const ref
        std::ostringstream oss;
        oss << spvReflectToYaml;
        SPDLOG_INFO("ShaderModule: {}", oss.str());
    }

    uint32_t descriptorSetCount = 0;
    reflectResult = reflectionModule->EnumerateDescriptorSets(&descriptorSetCount, nullptr);
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);
    std::vector<SpvReflectDescriptorSet *> reflectDescriptorSets(descriptorSetCount);
    reflectResult = reflectionModule->EnumerateDescriptorSets(&descriptorSetCount, std::data(reflectDescriptorSets));
    INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "EnumerateDescriptorSets returned {}", reflectResult);

    descriptorSetLayouts.reserve(descriptorSetCount);
    descriptorSetLayoutCreateInfos.reserve(descriptorSetCount);
    for (uint32_t index = 0; index < descriptorSetCount; ++index) {
        const auto reflectDecriptorSet = reflectDescriptorSets[index];
        INVARIANT(reflectDecriptorSet, "reflectDecriptorSet is null at #{}", index);

        auto & descriptorSetLayout = descriptorSetLayouts.emplace_back();

        descriptorSetLayout.set = reflectDecriptorSet->set;

        auto bindingCount = reflectDecriptorSet->binding_count;
        descriptorSetLayout.bindings.reserve(bindingCount);
        for (uint32_t b = 0; b < bindingCount; ++b) {
            const auto reflectDescriptorBinding = reflectionModule->GetDescriptorBinding(b, index, &reflectResult);
            INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "GetDescriptorBinding returned {} for binding #{} set #{}", reflectResult, b, index);
            INVARIANT(reflectDescriptorBinding, "");
            auto & descriptorSetLayoutBinding = descriptorSetLayout.bindings.emplace_back();

            descriptorSetLayoutBinding.binding = reflectDescriptorBinding->binding;

            descriptorSetLayoutBinding.descriptorType = spvReflectDescriiptorTypeToVk(reflectDescriptorBinding->descriptor_type);

            descriptorSetLayoutBinding.descriptorCount = 1;
            for (uint32_t d = 0; d < reflectDescriptorBinding->array.dims_count; ++d) {
                descriptorSetLayoutBinding.descriptorCount *= reflectDescriptorBinding->array.dims[d];
            }

            descriptorSetLayoutBinding.stageFlags = shaderStage;
        }
        auto & descriptorSetLayoutCreateInfo = descriptorSetLayoutCreateInfos.emplace_back();
        descriptorSetLayoutCreateInfo.flags = {};
        descriptorSetLayoutCreateInfo.setBindings(descriptorSetLayout.bindings);
    }

    auto pushConstantBlockCount = reflectionModule->GetShaderModule().push_constant_block_count;
    pushConstantRanges.reserve(pushConstantBlockCount);
    for (uint32_t index = 0; index < pushConstantBlockCount; ++index) {
        const auto reflectionPushConstantBlock = reflectionModule->GetPushConstantBlock(index, &reflectResult);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "GetPushConstantBlock returned {} for push constant block #{}", reflectResult, index);
        INVARIANT(reflectionPushConstantBlock, "");
        vk::PushConstantRange pushConstantRange = {
            .stageFlags = shaderStage,
            .offset = reflectionPushConstantBlock->offset,
            .size = reflectionPushConstantBlock->size,
        };
        pushConstantRanges.push_back(pushConstantRange);
    }

    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    pipelineVertexInputStateCreateInfo.flags = {};
    pipelineVertexInputStateCreateInfo.setVertexAttributeDescriptions(nullptr);
    pipelineVertexInputStateCreateInfo.setVertexBindingDescriptions(nullptr);

    auto inputVariableCount = reflectionModule->GetShaderModule().input_variable_count;
    for (uint32_t location = 0; location < inputVariableCount; ++location) {
        auto inputVariable = reflectionModule->GetInputVariableByLocation(location, &reflectResult);
        INVARIANT(reflectResult == SPV_REFLECT_RESULT_SUCCESS, "GetInputVariableByLocation returned {} for location #{}", reflectResult, location);
        INVARIANT(inputVariable, "");
    }
}

ShaderStages::ShaderStages(const Engine & engine) : engine{engine}, library{*engine.library}, device{*engine.device}
{}

void ShaderStages::append(const ShaderModule & shaderModule, std::string_view entryPoint)
{
    entryPoints.emplace_back(entryPoint);
    const auto & name = names.emplace_back(fmt::format("{}:{}", shaderModule.name, entryPoint));

    shaderStages.resize(shaderStages.size() + 1);
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
}

}  // namespace engine
