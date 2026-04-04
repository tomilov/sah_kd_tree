#include <compute/assert.hpp>
#include <compute/compute.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/fd.hpp>
#include <utils/math.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <iterator>
#include <utility>

#include <cstring>

template struct utils::OneTime<compute::DeviceMemory>::CheckTraits;
template struct utils::OneTime<compute::CudaStream>::CheckTraits;
template struct utils::OneTime<compute::MappedDeviceMemory>::CheckTraits;
template struct utils::OneTime<compute::CudaFileDriver>::CheckTraits;
template struct utils::OneTime<compute::CudaFile>::CheckTraits;
template struct utils::OneTime<compute::CudaFileReader>::CheckTraits;

namespace compute
{

namespace
{

// Win32 CU_MEM_HANDLE_TYPE_WIN32
constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

::CUmemAllocationProp makeMemAllocationProp(::CUdevice cuDev)
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

size_t getAllocationGranularity(
    const ::CUmemAllocationProp & memAllocationPropIn,
    CUmemAllocationGranularity_flags_enum memAllocationGranularityFlag)
{
    size_t allocGranularityOut = 0;
    CU_CALL_CHECK(cuMemGetAllocationGranularity, &allocGranularityOut, &memAllocationPropIn, memAllocationGranularityFlag);
    SPDLOG_INFO("{} alloc granularity {}", memAllocationGranularityFlag, allocGranularityOut);
    return allocGranularityOut;
}

size_t getAlignedAllocationSize(
    const ::CUmemAllocationProp & memAllocationPropIn,
    size_t allocationSize,
    size_t allocationAlignment)
{
    const size_t recommendedAllocGranularity = getAllocationGranularity(memAllocationPropIn, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    return utils::alignUp(std::max(allocationSize, allocationAlignment), recommendedAllocGranularity);
}

::CUmemGenericAllocationHandle makeMemGenericAllocationHandle(
    size_t alignedAllocationSizeIn,
    const ::CUmemAllocationProp & memAllocationPropIn)
{
    ::CUmemGenericAllocationHandle allocationHandleOut = {};
    CU_CALL_CHECK(cuMemCreate, &allocationHandleOut, alignedAllocationSizeIn, &memAllocationPropIn, 0);
    return allocationHandleOut;
}

::CUmemGenericAllocationHandle importMemGenericAllocationHandle(utils::Fd && fd)
{
    ::CUmemGenericAllocationHandle allocationHandleOut = {};
    CU_CALL_CHECK(cuMemImportFromShareableHandle, &allocationHandleOut, utils::autoCast(fd.getFd()), kHandleType);
    return allocationHandleOut;
}

int selectCudaDevice(
    cudaDeviceProp & devProp,
    const DeviceUuidType & deviceUuid)
{
    int devCount = 0;
    CUDA_CALL_CHECK(cudaGetDeviceCount, &devCount);
    for (int cudaDev = 0; cudaDev < devCount; ++cudaDev) {
        CUDA_CALL_CHECK(cudaGetDeviceProperties, &devProp, cudaDev);
        if (std::bit_cast<DeviceUuidType>(devProp.uuid) == deviceUuid) {
            return cudaDev;
        }
    }
    return cudaInvalidDeviceId;
}

::CUdevice findCuDevice(const cudaDeviceProp & devProp)
{
    ::CUdevice cuDev = CU_DEVICE_INVALID;
    ::CUuuid uuid = {};
    int cuDevCount = 0;
    CU_CALL_CHECK(cuDeviceGetCount, &cuDevCount);
    for (int cuDevIndex = 0; cuDevIndex < cuDevCount; ++cuDevIndex) {
        CU_CALL_CHECK(cuDeviceGet, &cuDev, cuDevIndex);
        CU_CALL_CHECK(cuDeviceGetUuid, &uuid, cuDev);
        if (std::bit_cast<DeviceUuidType>(devProp.uuid) == std::bit_cast<DeviceUuidType>(uuid)) {
            {
                int deviceAttribute = 0;
                CU_CALL_CHECK(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev);
                SKT_INVARIANT(deviceAttribute == CU_COMPUTEMODE_DEFAULT, "{}", deviceAttribute);
            }
            {
                int deviceAttribute = 0;
                CU_CALL_CHECK(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDev);
                SKT_INVARIANT(deviceAttribute != 0, "Virtual address management is not supported");
            }
            {
                int deviceAttribute = 0;
                // Win32: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
                CU_CALL_CHECK(cuDeviceGetAttribute, &deviceAttribute, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cuDev);
                SKT_INVARIANT(deviceAttribute != 0, "Posix file descriptor handle type is not supported");
            }
            return cuDev;
        }
    }
    return CU_DEVICE_INVALID;
}

}  // namespace

MappedDeviceMemory::MappedDeviceMemory(
    const ::CUmemLocation & location,
    size_t allocGranularity,
    size_t alignedAllocationSizeIn,
    ::CUmemGenericAllocationHandle allocationHandle)
    : alignedAllocationSize{alignedAllocationSizeIn}
{
    CU_CALL_CHECK(cuMemAddressReserve, &devPtr, alignedAllocationSize, allocGranularity, devPtr, 0);
    CU_CALL_CHECK(cuMemMap, devPtr, alignedAllocationSize, 0, allocationHandle, 0);
    ::CUmemAccessDesc accessDescriptor[] = {
        {
            .location = location,
            .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        },
    };
    CU_CALL_CHECK(cuMemSetAccess, devPtr, alignedAllocationSize, std::data(accessDescriptor), std::size(accessDescriptor));
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
    CU_CALL_CHECK(cuMemUnmap, devPtr, alignedAllocationSize);
    CU_CALL_CHECK(cuMemAddressFree, devPtr, alignedAllocationSize);
}

DeviceMemory::DeviceMemory(DeviceMemory && rhs) noexcept
    : memAllocationProp{rhs.memAllocationProp}
    , allocMinimumGranularity{rhs.allocMinimumGranularity}
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
    CU_CALL_CHECK(cuMemRelease, allocationHandle);  // after both cuMemExportToShareableHandle and cuMemMap
}

MappedDeviceMemory DeviceMemory::map() const &
{
    return {memAllocationProp.location, allocMinimumGranularity, alignedAllocationSize, allocationHandle};
}

std::optional<utils::Fd> DeviceMemory::exportMemoryObject() const
{
    int fd = -1;
    CU_CALL_CHECK(cuMemExportToShareableHandle, &fd, allocationHandle, kHandleType, 0);
    if (fd < 0) {
        return std::nullopt;
    }
    return utils::Fd{fd};
}

DeviceMemory::DeviceMemory(
    ::CUdevice cuDev,
    size_t allocationSize,
    size_t allocationAlignment)
    : memAllocationProp{makeMemAllocationProp(cuDev)}
    , allocMinimumGranularity{getAllocationGranularity(
          memAllocationProp,
          CU_MEM_ALLOC_GRANULARITY_MINIMUM)}
    , alignedAllocationSize{getAlignedAllocationSize(
          memAllocationProp,
          allocationSize,
          allocationAlignment)}
    , allocationHandle{makeMemGenericAllocationHandle(
          alignedAllocationSize,
          memAllocationProp)}
{}

DeviceMemory::DeviceMemory(
    ::CUdevice cuDev,
    utils::Fd && fd,
    size_t allocationSize,
    size_t allocationAlignment)
    : memAllocationProp{makeMemAllocationProp(cuDev)}
    , allocMinimumGranularity{getAllocationGranularity(
          memAllocationProp,
          CU_MEM_ALLOC_GRANULARITY_MINIMUM)}
    , alignedAllocationSize{getAlignedAllocationSize(
          memAllocationProp,
          allocationSize,
          allocationAlignment)}
    , allocationHandle{importMemGenericAllocationHandle(std::move(fd))}
{}

CudaDevice::CudaDevice(const DeviceUuidType & deviceUuid)
{
    cudaDev = selectCudaDevice(devProp, deviceUuid);
    SKT_INVARIANT(cudaDev != cudaInvalidDeviceId, "No matching by UUID devices found using CUDA Runtime API");
    cuDev = findCuDevice(devProp);
    SKT_INVARIANT(cuDev != CU_DEVICE_INVALID, "No matching by UUID devices found using CUDA Driver API");
}

bool CudaDevice::operator==(const CudaDevice & rhs) const noexcept
{
    return std::bit_cast<DeviceUuidType>(devProp.uuid) == std::bit_cast<DeviceUuidType>(rhs.devProp.uuid);
}

int CudaDevice::getDeviceCount()
{
    int devCount = 0;
    CUDA_CALL_CHECK(cudaGetDeviceCount, &devCount);
    return devCount;
}

CudaDevice CudaDevice::getInvalidDevice()
{
    return {};
}

CudaDevice CudaDevice::getCurrentDevice()
{
    int cudaDev = cudaInvalidDeviceId;
    CUDA_CALL_CHECK(cudaGetDevice, &cudaDev);
    return CudaDevice{cudaDev};
}

CudaDevice CudaDevice::chooseDevice()
{
    cudaDeviceProp devProp = {};
    devProp.canMapHostMemory = 1;
    int cudaDev = cudaInvalidDeviceId;
    CUDA_CALL_CHECK(cudaChooseDevice, &cudaDev, &devProp);
    return CudaDevice{cudaDev};
}

void CudaDevice::setCurrentDevice() const
{
    CUDA_CALL_CHECK(cudaSetDevice, cudaDev);
}

void CudaDevice::synchronize() const
{
    CUDA_CALL_CHECK(cudaDeviceSynchronize);
}

DeviceMemory CudaDevice::makeDeviceMemory(
    size_t allocationSize,
    size_t allocationAlignment) const &
{
    return {cuDev, allocationSize, allocationAlignment};
}

DeviceMemory CudaDevice::makeDeviceMemory(
    utils::Fd && fd,
    size_t allocationSize,
    size_t allocationAlignment) const &
{
    return {cuDev, std::move(fd), allocationSize, allocationAlignment};
}

size_t CudaDevice::getMaxPitch() const
{
    int maxPitch = 0;
    CU_CALL_CHECK(cuDeviceGetAttribute, &maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, cuDev);
    return utils::autoCast(maxPitch);
}

CudaDevice::CudaDevice(int cudaDevIn)
    : cudaDev{cudaDevIn}
{
    CUDA_CALL_CHECK(cudaGetDeviceProperties, &devProp, cudaDev);
    cuDev = findCuDevice(devProp);
    SKT_INVARIANT(cuDev != CU_DEVICE_INVALID, "No matching by UUID devices found using CUDA Driver API");
}

CudaStream::CudaStream(cudaStream_t cudaStreamIn)
    : cudaStream{cudaStreamIn}
{}

CudaStream::CudaStream(CudaStream && rhs) noexcept
    : cudaStream{std::exchange(
          rhs.cudaStream,
          nullptr)}
{}

CudaStream::~CudaStream()
{
    if (cudaStream == nullptr) {
        return;
    }
    synchronize();
    if (cudaStream == cudaStreamPerThread) {
        return;
    }
    CUDA_CALL_CHECK(cudaStreamDestroy, cudaStream);
}

cudaStream_t CudaStream::getHandle() const &
{
    return cudaStream;
}

CudaStream CudaStream::makeDefault()
{
    cudaStream_t cudaStream;
    CUDA_CALL_CHECK(cudaStreamCreate, &cudaStream);
    return CudaStream{cudaStream};
}

CudaStream CudaStream::makePerThread()
{
    return CudaStream{cudaStreamPerThread};
}

CudaStream CudaStream::makeNonBlocking()
{
    cudaStream_t cudaStream = nullptr;
    CUDA_CALL_CHECK(cudaStreamCreateWithFlags, &cudaStream, cudaStreamNonBlocking);
    return CudaStream{cudaStream};
}

void CudaStream::synchronize() const
{
    CUDA_CALL_CHECK(cudaStreamSynchronize, cudaStream);
}

void CudaStream::synchronize(cudaStream_t cudaStreamIn)
{
    CUDA_CALL_CHECK(cudaStreamSynchronize, cudaStreamIn);
}

CudaFileDriver::CudaFileDriver() noexcept
{
    CUFILE_CALL_CHECK(cuFileDriverOpen);
}

CudaFileDriver::~CudaFileDriver()
{
    CUFILE_CALL_CHECK(cuFileDriverClose);
}

CudaFile CudaFileDriver::createFile(utils::Fd && fd) const &  // NOLINT: readability-convert-member-functions-to-static
{
    return CudaFile{std::move(fd)};
}

CudaFileReader::CudaFileReader()
{
    // TODO:
}

std::optional<size_t> CudaFile::readAsync(
    intptr_t /*fileOffset*/,
    const DeviceMemory & /*deviceMemory*/,
    size_t /*size*/,
    intptr_t /*memOffset*/,
    const CudaStream & /*stream*/)
{
    // cuFileReadAsync
    return std::nullopt;
}

auto CudaFile::makeFileHandle(int fd) -> CUfileHandle_t
{
    CUfileDescr_t fileDescr = {};
    fileDescr.handle.fd = fd;
    fileDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t fileHandleOut = {};
    CUFILE_CALL_CHECK(cuFileHandleRegister, &fileHandleOut, &fileDescr);
    return fileHandleOut;
}

CudaFile::CudaFile(utils::Fd && fdIn)
    : fd{std::move(fdIn)}
    , fileHandle{&cuFileHandleDeregister,
          makeFileHandle(fd.getFd())}
{}

}  // namespace compute
