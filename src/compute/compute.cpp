#include <compute/assert.hpp>
#include <compute/compute.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/fd.hpp>
#include <utils/math.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <utility>

#include <cstring>

namespace compute
{

MappedDeviceMemory::MappedDeviceMemory(
    const ::CUmemLocation & location,
    size_t allocGranularity,
    size_t alignedAllocationSizeIn,
    ::CUmemGenericAllocationHandle allocationHandle)
    : alignedAllocationSize{alignedAllocationSizeIn}
{
    CU_CALL(cuMemAddressReserve, &devPtr, alignedAllocationSize, allocGranularity, devPtr, 0);
    CU_CALL(cuMemMap, devPtr, alignedAllocationSize, 0, allocationHandle, 0);
    ::CUmemAccessDesc accessDescriptor[] = {
        {
            .location = location,
            .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        },
    };
    CU_CALL(cuMemSetAccess, devPtr, alignedAllocationSize, std::data(accessDescriptor), std::size(accessDescriptor));
}

MappedDeviceMemory::MappedDeviceMemory(MappedDeviceMemory && rhs) noexcept
    : alignedAllocationSize{rhs.alignedAllocationSize}
    , devPtr{std::exchange(
          rhs.devPtr,
          ::CUdeviceptr{})}
{}

MappedDeviceMemory::~MappedDeviceMemory()
{
    if (devPtr == ::CUdeviceptr{}) {
        return;
    }
    CU_CALL(cuMemUnmap, devPtr, alignedAllocationSize);
    CU_CALL(cuMemAddressFree, devPtr, alignedAllocationSize);
}

DeviceMemory::DeviceMemory(
    ::CUdevice cuDev,
    size_t allocationSize,
    size_t allocationAlignment)
    : memAllocationProp{makeMemAllocationProp(cuDev)}
    , allocGranularity{getAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)}
    , alignedAllocationSize{getAlignedAllocationSize(
          allocationSize,
          allocationAlignment)}
    , allocationHandle{makeMemGenericAllocationHandle()}
{}

DeviceMemory::DeviceMemory(
    ::CUdevice cuDev,
    utils::Fd fd,
    size_t allocationSize,
    size_t allocationAlignment)
    : memAllocationProp{makeMemAllocationProp(cuDev)}
    , allocGranularity{getAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)}
    , alignedAllocationSize{getAlignedAllocationSize(
          allocationSize,
          allocationAlignment)}
    , allocationHandle{importMemGenericAllocationHandle(std::move(fd))}
{}

DeviceMemory::DeviceMemory(DeviceMemory && rhs) noexcept
    : memAllocationProp{rhs.memAllocationProp}
    , allocGranularity{rhs.allocGranularity}
    , alignedAllocationSize{rhs.alignedAllocationSize}
    , allocationHandle{std::exchange(
          rhs.allocationHandle,
          ::CUmemGenericAllocationHandle{})}
{}

DeviceMemory::~DeviceMemory()
{
    if (allocationHandle == ::CUmemGenericAllocationHandle{}) {
        return;
    }
    CU_CALL(cuMemRelease, allocationHandle);  // after both cuMemExportToShareableHandle and cuMemMap
}

MappedDeviceMemory DeviceMemory::map() const &
{
    return {memAllocationProp.location, allocGranularity, alignedAllocationSize, allocationHandle};
}

utils::Fd DeviceMemory::exportMemoryObject() const
{
    int fd = -1;
    CU_CALL(cuMemExportToShareableHandle, &fd, allocationHandle, kHandleType, 0);
    return utils::Fd{fd};
}

::CUmemAllocationProp DeviceMemory::makeMemAllocationProp(::CUdevice cuDev)
{
    return {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = kHandleType,
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = cuDev,
        },
        .win32HandleMetaData = nullptr,  // Win32 Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
        .allocFlags = {},
    };
}

size_t DeviceMemory::getAllocationGranularity(CUmemAllocationGranularity_flags_enum memAllocationGranularityFlag) const
{
    size_t allocGranularityOut = 0;
    CU_CALL(cuMemGetAllocationGranularity, &allocGranularityOut, &memAllocationProp, memAllocationGranularityFlag);
    const char * kind = nullptr;
    switch (memAllocationGranularityFlag) {
    case CU_MEM_ALLOC_GRANULARITY_MINIMUM: {
        kind = "minimum";
        break;
    }
    case CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: {
        kind = "recommended";
        break;
    }
    }
    INVARIANT(kind, "{}", fmt::underlying(memAllocationGranularityFlag));
    SPDLOG_INFO("{} allocGranularity {}", kind, allocGranularityOut);
    return allocGranularityOut;
}

size_t DeviceMemory::getAlignedAllocationSize(
    size_t allocationSize,
    size_t allocationAlignment) const
{
    const size_t recommendedAllocGranularity = getAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    return utils::alignUp(std::max(allocationSize, allocationAlignment), recommendedAllocGranularity);
}

::CUmemGenericAllocationHandle DeviceMemory::makeMemGenericAllocationHandle() const
{
    ::CUmemGenericAllocationHandle allocationHandleOut = {};
    const auto result = cuMemCreate(&allocationHandleOut, alignedAllocationSize, &memAllocationProp, 0);
    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        throw OutOfMemoryException{};
    }
    INVARIANT(result == CUDA_SUCCESS, "{}", result);
    return allocationHandleOut;
}

::CUmemGenericAllocationHandle DeviceMemory::importMemGenericAllocationHandle(utils::Fd fd)
{
    ::CUmemGenericAllocationHandle allocationHandleOut = {};
    const auto result = cuMemImportFromShareableHandle(&allocationHandleOut, utils::autoCast(fd.getFd()), kHandleType);
    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        throw OutOfMemoryException{};
    }
    INVARIANT(result == CUDA_SUCCESS, "{}", result);
    return allocationHandleOut;
}

CudaDevice::CudaDevice(const std::optional<DeviceUuidType> & deviceUuidIn)
    : deviceUuid{deviceUuidIn}
{
    cudaDeviceProp devProp = {};
    static_assert(sizeof(DeviceUuidType) == sizeof devProp.uuid);
    {
        int devCount = 0;
        CUDA_CALL(cudaGetDeviceCount, &devCount);
        INVARIANT(devCount > 0, "");
        for (cudaDev = 0; cudaDev < devCount; ++cudaDev) {
            CUDA_CALL(cudaGetDeviceProperties, &devProp, cudaDev);
            if (!deviceUuid || (std::memcmp(&devProp.uuid, std::data(deviceUuid.value()), sizeof devProp.uuid) == 0)) {
                break;
            }
        }
        INVARIANT(cudaDev != devCount, "No matching by UUID devices found using CUDA Runtime API");
    }
    {
        int cuDevCount = 0;
        CU_CALL(cuDeviceGetCount, &cuDevCount);
        int cuDevIndex = 0;
        for (; cuDevIndex < cuDevCount; ++cuDevIndex) {
            CU_CALL(cuDeviceGet, &cuDev, cuDevIndex);
            ::CUuuid uuid = {};
            CU_CALL(cuDeviceGetUuid, &uuid, cuDev);
            static_assert(sizeof devProp.uuid == sizeof uuid);
            if (std::memcmp(&devProp.uuid, &uuid, sizeof uuid) == 0) {
                break;
            }
        }
        INVARIANT(cuDevIndex != cuDevCount, "No matching by UUID devices found using CUDA Driver API");
    }
    {
        int deviceAttribute = 0;
        CU_CALL(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev);
        INVARIANT(deviceAttribute == CU_COMPUTEMODE_DEFAULT, "{}", deviceAttribute);
    }
    {
        int deviceAttribute = 0;
        CU_CALL(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDev);
        INVARIANT(deviceAttribute != 0, "Virtual address management is not supported");
    }
    {
        int deviceAttribute = 0;
        // Win32: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
        CU_CALL(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cuDev);
        INVARIANT(deviceAttribute != 0, "Posix file descriptor handle type is not supported");
    }
}

void CudaDevice::setCurrentDevice() const
{
    CUDA_CALL(cudaSetDevice, getCudaRuntimeDev());
}

CudaStream::CudaStream()
{
    CUDA_CALL(cudaStreamCreate, &cudaStream);
}

CudaStream::~CudaStream()
{
    CUDA_CALL(cudaStreamDestroy, cudaStream);
}

void CudaStream::synchronize() const
{
    CUDA_CALL(cudaStreamSynchronize, cudaStream);
}

auto CudaFile::makeFileHandle(int fd) -> CUfileHandle_t
{
    CUfileDescr_t fileDescr = {};
    fileDescr.handle.fd = fd;
    fileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t fileHandleOut = {};
    CUFILE_CALL(cuFileHandleRegister, &fileHandleOut, &fileDescr);
    return fileHandleOut;
}

CudaFile::CudaFile(utils::Fd && fdIn)
    : fd{std::move(fdIn)}
    , fileHandle{&cuFileHandleDeregister,
          makeFileHandle(fd.getFd())}
{}

CudaFileDriver::CudaFileDriver()
{
    CUFILE_CALL(cuFileDriverOpen);
}

CudaFileDriver::~CudaFileDriver()
{
    CUFILE_CALL(cuFileDriverClose);
}

CudaFile CudaFileDriver::createFile(utils::Fd && fd)
{
    return CudaFile{std::move(fd)};
}

}  // namespace compute
