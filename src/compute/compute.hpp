#pragma once

#include <compute/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>

#include <fmt/format.h>

#include <new>
#include <optional>

#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>

#include <compute/compute_export.h>

#define CU_CHECK_ERROR(call)                                        \
    do {                                                            \
        ::CUresult result = CUDA_SUCCESS;                           \
        INVARIANT((result = (call)) == CUDA_SUCCESS, "{}", result); \
    } while (false)

#define CUDA_CHECK_ERROR(call)                                   \
    do {                                                         \
        cudaError error = cudaSuccess;                           \
        INVARIANT((error = (call)) == cudaSuccess, "{}", error); \
    } while (false)

template<>
struct fmt::formatter<cudaError> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(cudaError error, FormatContext & ctx) const
    {
        const char * errorName = ::cudaGetErrorName(error);
        const char * errorString = ::cudaGetErrorString(error);
        return fmt::format_to(ctx.out(), "{}: {}", errorName, errorString);
    }
};

template<>
struct fmt::formatter<::CUresult> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(::CUresult result, FormatContext & ctx) const
    {
        const char * errorName = "unknown";
        const char * errorString = "unknown";
        ::cuGetErrorName(result, &errorName);
        ::cuGetErrorString(result, &errorString);
        return fmt::format_to(ctx.out(), "{}: {}", errorName, errorString);
    }
};

namespace compute
{

class COMPUTE_EXPORT OutOfMemoryException : public std::bad_alloc
{
    [[nodiscard]] const char * what() const noexcept override
    {
        return "OutOfMemoryException: out of memory";
    }
};

class COMPUTE_EXPORT MappedDeviceMemory : utils::OneTime<MappedDeviceMemory>
{
public:
    MappedDeviceMemory(const ::CUmemLocation & location, size_t allocGranularity, size_t alignedAllocationSize, ::CUmemGenericAllocationHandle allocationHandle);
    MappedDeviceMemory(MappedDeviceMemory && rhs) noexcept;
    ~MappedDeviceMemory();

    [[nodiscard]] ::CUdeviceptr getPtr() const &
    {
        return devPtr;
    }

private:
    const size_t alignedAllocationSize;

    ::CUdeviceptr devPtr = {};

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT DeviceMemory : utils::OneTime<DeviceMemory>
{
public:
    DeviceMemory(::CUdevice cuDev, size_t allocationSize, size_t allocationAlignment = 0);
    DeviceMemory(::CUdevice cuDev, utils::Fd fd, size_t allocationSize, size_t allocationAlignment = 0);
    DeviceMemory(DeviceMemory && rhs) noexcept;
    ~DeviceMemory();

    [[nodiscard]] MappedDeviceMemory map() const &;
    [[nodiscard]] utils::Fd exportMemoryObject() const;

    [[nodiscard]] size_t getSize() const
    {
        return alignedAllocationSize;
    }

private:
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    const ::CUmemAllocationProp memAllocationProp;
    const size_t allocGranularity;
    const size_t alignedAllocationSize;

    ::CUmemGenericAllocationHandle allocationHandle = {};

    static ::CUmemAllocationProp makeMemAllocationProp(::CUdevice cuDev);
    [[nodiscard]] size_t getAllocationGranularity(CUmemAllocationGranularity_flags_enum memAllocationGranularityFlag) const;
    [[nodiscard]] size_t getAlignedAllocationSize(size_t allocationSize, size_t allocationAlignment) const;
    [[nodiscard]] ::CUmemGenericAllocationHandle makeMemGenericAllocationHandle() const;
    [[nodiscard]] ::CUmemGenericAllocationHandle importMemGenericAllocationHandle(utils::Fd fd) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT CudaDevice
{
public:
    explicit CudaDevice(const std::optional<DeviceUuidType> & deviceUuid);

    [[nodiscard]] const std::optional<DeviceUuidType> & getDeviceUuid() const &
    {
        return deviceUuid;
    }

    [[nodiscard]] int getCudaRuntimeDev() const &
    {
        return cudaDev;
    }

    [[nodiscard]] ::CUdevice getCudaDriverDev() const &
    {
        return cuDev;
    }

private:
    const std::optional<DeviceUuidType> deviceUuid;

    int cudaDev = cudaInvalidDeviceId;
    ::CUdevice cuDev = CU_DEVICE_INVALID;
};

}  // namespace compute
