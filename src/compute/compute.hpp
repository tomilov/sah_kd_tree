#pragma once

#include <compute/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>
#include <utils/scope_guard.hpp>

#include <optional>

#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <compute/compute_export.h>

namespace compute
{

class COMPUTE_EXPORT CudaDevice
{
public:
    explicit CudaDevice(const DeviceUuidType & deviceUuid);

    [[nodiscard]] bool operator==(const CudaDevice & rhs) const noexcept;

    [[nodiscard]] static int getDeviceCount() COMPUTE_EXPORT;

    static CudaDevice getInvalidDevice() COMPUTE_EXPORT;
    static CudaDevice getCurrentDevice() COMPUTE_EXPORT;
    static CudaDevice chooseDevice() COMPUTE_EXPORT;

    void setCurrentDevice() const;
    static void synchronize();

    [[nodiscard]] DeviceMemory makeDeviceMemory(
        size_t allocationSize,
        size_t allocationAlignment = 0) const &;

    [[nodiscard]] DeviceMemory makeDeviceMemory(
        utils::Fd && fd,
        size_t allocationSize,
        size_t allocationAlignment = 0) const &;

private:
    friend CudaDevice getInvalidDevice();
    friend CudaDevice getCurrentDevice();
    friend CudaDevice chooseDevice();

    cudaDeviceProp devProp = {};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
    int cudaDev = cudaInvalidDeviceId;
    ::CUdevice cuDev = CU_DEVICE_INVALID;
#pragma GCC diagnostic pop

    CudaDevice() = default;
    explicit CudaDevice(int cudaDevIn);
};

class COMPUTE_EXPORT CudaStream : utils::Copyable<CudaStream>
{
public:
    CudaStream();
    CudaStream(const CudaStream &) noexcept = default;
    CudaStream & operator=(const CudaStream &) noexcept = default;
    ~CudaStream();  // quote: "Note that destroying a stream is an asynchronous operation"

    [[nodiscard]] bool operator==(const CudaStream & rhs) const noexcept
    {
        return cudaStream == rhs.cudaStream;
    }

    [[nodiscard]] cudaStream_t getHandle() const &
    {
        return cudaStream;
    }

    void synchronize() const;

private:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
    cudaStream_t cudaStream = cudaStreamPerThread;
#pragma GCC diagnostic pop

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT DeviceMemory : utils::OneTime<DeviceMemory>
{
public:
    DeviceMemory(DeviceMemory && rhs) noexcept;
    ~DeviceMemory();

    [[nodiscard]] MappedDeviceMemory map() const &;
    [[nodiscard]] std::optional<utils::Fd> exportMemoryObject() const;

    [[nodiscard]] size_t getSize() const
    {
        return alignedAllocationSize;
    }

private:
    friend CudaDevice;

    const ::CUmemAllocationProp memAllocationProp;
    const size_t allocMinimumGranularity;
    const size_t alignedAllocationSize;

    ::CUmemGenericAllocationHandle allocationHandle = {};

    DeviceMemory(
        ::CUdevice cuDev,
        size_t allocationSize,
        size_t allocationAlignment);

    DeviceMemory(
        ::CUdevice cuDev,
        utils::Fd && fd,
        size_t allocationSize,
        size_t allocationAlignment);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT MappedDeviceMemory : utils::OneTime<MappedDeviceMemory>
{
public:
    MappedDeviceMemory(
        const ::CUmemLocation & location,
        size_t allocGranularity,
        size_t alignedAllocationSize,
        ::CUmemGenericAllocationHandle allocationHandle);
    MappedDeviceMemory(MappedDeviceMemory && rhs) noexcept;
    ~MappedDeviceMemory();

    [[nodiscard]] ::CUdeviceptr getCuDevPtr() const &
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

class COMPUTE_EXPORT CudaFileDriver : utils::OneTime<CudaFileDriver>
{
public:
    CudaFileDriver() noexcept;
    CudaFileDriver(CudaFileDriver &&) noexcept = default;
    ~CudaFileDriver();

    CudaFile createFile(utils::Fd && fd) const &;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT CudaFile : utils::OneTime<CudaFile>
{
public:
    std::optional<size_t> readAsync(
        intptr_t fileOffset,
        const DeviceMemory & deviceMemory,
        size_t size,
        intptr_t memOffset,
        const CudaStream & stream);

private:
    friend CudaFileDriver;

    using FileHolder = utils::ScopeGuard<decltype(&cuFileHandleDeregister), CUfileHandle_t>;

    utils::Fd fd;
    FileHolder fileHandle;

    static CUfileHandle_t makeFileHandle(int fd);

    explicit CudaFile(utils::Fd && fd);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class COMPUTE_EXPORT CudaFileReader : utils::OneTime<CudaFile>
{
public:
    CudaFileReader();

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace compute
