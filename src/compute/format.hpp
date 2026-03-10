#pragma once

#include <compute/fwd.hpp>
#include <utils/pp.hpp>

#include <fmt/format.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <compute/compute_export.h>

template<>
struct fmt::formatter<cudaError> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        cudaError error,
        FormatContext & ctx) const
    {
        const char * errorName = ::cudaGetErrorName(error);
        const char * errorString = ::cudaGetErrorString(error);
        return fmt::format_to(ctx.out(), "{} ({})", errorName, errorString);
    }
};

template<>
struct fmt::formatter<::CUresult> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        ::CUresult result,
        FormatContext & ctx) const
    {
        const char * errorName = "unknown";
        const char * errorString = "unknown";
        ::cuGetErrorName(result, &errorName);
        ::cuGetErrorString(result, &errorString);
        return fmt::format_to(ctx.out(), "{} ({})", errorName, errorString);
    }
};

template<>
struct fmt::formatter<CUfileError_t> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        CUfileError_t status,
        FormatContext & ctx) const
    {
        constexpr auto abs = [](CUfileOpError opError) -> CUfileOpError
        {
            return opError;
        };
        return fmt::format_to(ctx.out(), "{}, {}", CUFILE_ERRSTR(status.err), CU_FILE_CUDA_ERR(status));
    }
};

namespace compute
{

}  // namespace compute
