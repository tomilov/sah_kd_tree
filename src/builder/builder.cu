#include <builder/builder.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/math.hpp>

#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/pointer.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>

#include <cstddef>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define CUDA_CHECK_ERROR(call)                                   \
    do {                                                         \
        cudaError error = cudaSuccess;                           \
        INVARIANT((error = (call)) == cudaSuccess, "{}", error); \
    } while (false)
#define CU_CHECK_ERROR(call)                                        \
    do {                                                            \
        ::CUresult result = CUDA_SUCCESS;                           \
        INVARIANT((result = (call)) == CUDA_SUCCESS, "{}", result); \
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

namespace builder
{

namespace
{
#if SAH_KD_TREE_HEADER_ONLY
struct Traits  // cannot be member typedef of Tree::Impl because of wierd CUDA parser
{
    using sah_kd_tree::DefaultTraits::F;
    using sah_kd_tree::DefaultTraits::I;
    using sah_kd_tree::DefaultTraits::U;
    using MemoryResource = thrust::device_memory_resource;
    template<typename T>
    using Allocator = thrust::mr::allocator<T, MemoryResource>;
    template<typename T>
    using Vector = thrust::device_vector<T, Allocator<T>>;
};
#else
using Traits = sah_kd_tree::DefaultTraits;
#endif
}  // namespace

class CudaDevice : utils::OneTime<CudaDevice>
{
public:
    CudaDevice(const std::optional<DeviceUuidType> & deviceUuid)
        : deviceUuid{deviceUuid}
    {
        checkTraits();
        selectCudaDevice();
    }

    int getCudaDev() const
    {
        return cudaDev;
    }

    ::CUdevice getCuDev() const
    {
        return cuDev;
    }

private:
    const std::optional<DeviceUuidType> & deviceUuid;

    int cudaDev = cudaInvalidDeviceId;
    ::CUdevice cuDev = CU_DEVICE_INVALID;

    void selectCudaDevice()
    {
        cudaDeviceProp devProp = {};
        static_assert(sizeof(DeviceUuidType) == sizeof cudaDeviceProp::uuid);
        CUDA_CHECK_ERROR(cudaGetDevice(&cudaDev));
        if (cudaDev == cudaInvalidDeviceId) {
            int devCount = 0;
            CUDA_CHECK_ERROR(cudaGetDeviceCount(&devCount));
            INVARIANT(devCount > 0, "");
            for (cudaDev = 0; cudaDev < devCount; ++cudaDev) {
                CUDA_CHECK_ERROR(cudaGetDeviceProperties(&devProp, cudaDev));
                if (!deviceUuid || (std::memcmp(&devProp.uuid, std::data(deviceUuid.value()), sizeof devProp.uuid) == 0)) {
                    break;
                }
            }
            INVARIANT(cudaDev != devCount, "No matching by UUID devices found using CUDA Runtime API");
            selectCuDevice(devProp.uuid);
            CUDA_CHECK_ERROR(cudaSetDevice(cudaDev));
        } else {
            CUDA_CHECK_ERROR(cudaGetDeviceProperties(&devProp, cudaDev));
            selectCuDevice(devProp.uuid);
        }
    }

    void selectCuDevice(const cudaUUID_t & cudaDeviceUuid)
    {
        {
            int cuDevCount = 0;
            CU_CHECK_ERROR(cuDeviceGetCount(&cuDevCount));
            int cuDevIndex = 0;
            for (; cuDevIndex < cuDevCount; ++cuDevIndex) {
                CU_CHECK_ERROR(cuDeviceGet(&cuDev, cuDevIndex));
                ::CUuuid uuid = {};
                CU_CHECK_ERROR(cuDeviceGetUuid(&uuid, cuDev));
                static_assert(sizeof cudaDeviceUuid == sizeof uuid);
                if (std::memcmp(&cudaDeviceUuid, &uuid, sizeof uuid) == 0) {
                    break;
                }
            }
            INVARIANT(cuDevIndex != cuDevCount, "No matching by UUID devices found using CUDA Driver API");
        }
        {
            int deviceAttribute = 0;
            CU_CHECK_ERROR(cuDeviceGetAttribute(&deviceAttribute, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev));
            INVARIANT(deviceAttribute == CU_COMPUTEMODE_DEFAULT, "{}", deviceAttribute);
        }
        {
            int deviceAttribute = 0;
            CU_CHECK_ERROR(cuDeviceGetAttribute(&deviceAttribute, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDev));
            INVARIANT(deviceAttribute != 0, "Virtual address management is not supported");
        }
        {
            int deviceAttribute = 0;
            // Win32: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
            CU_CHECK_ERROR(cuDeviceGetAttribute(&deviceAttribute, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cuDev));
            INVARIANT(deviceAttribute != 0, "Posix file descriptor handle type is not supported");
        }
    }
};

class DeviceMemory
{
public:
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    class MappedDeviceMemory
    {
    public:
        MappedDeviceMemory(::CUmemGenericAllocationHandle allocationHandle, size_t alignedAllocationSize, size_t allocGranularity, const ::CUmemLocation & location)
            : allocationHandle{allocationHandle}
            , alignedAllocationSize{alignedAllocationSize}
            , allocGranularity{allocGranularity}
            , location{location}
        {
            CU_CHECK_ERROR(cuMemAddressReserve(&devPtr, alignedAllocationSize, allocGranularity, devPtr, 0));
            CU_CHECK_ERROR(cuMemMap(devPtr, alignedAllocationSize, 0, allocationHandle, 0));
            ::CUmemAccessDesc accessDescriptor[] = {
                {
                    .location = location,
                    .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                },
            };
            CU_CHECK_ERROR(cuMemSetAccess(devPtr, alignedAllocationSize, std::data(accessDescriptor), std::size(accessDescriptor)));
        }

        ~MappedDeviceMemory()
        {
            CU_CHECK_ERROR(cuMemUnmap(devPtr, alignedAllocationSize));
            CU_CHECK_ERROR(cuMemAddressFree(devPtr, alignedAllocationSize));
        }

        void * getPtr() const
        {
            return utils::autoCast(devPtr);
        }

    private:
        friend DeviceMemory;

        const ::CUmemGenericAllocationHandle allocationHandle;
        const size_t alignedAllocationSize;
        const size_t allocGranularity;
        const ::CUmemLocation & location;

        ::CUdeviceptr devPtr = {};
    };

    DeviceMemory(::CUdevice cuDev, size_t minAlignment, size_t allocationSize, size_t allocationAlignment)
        : cuDev{cuDev}
        , minAlignment{minAlignment}
        , allocationSize{allocationSize}
        , allocationAlignment{allocationAlignment}
    {
        init();
    }

    ~DeviceMemory()
    {
        CU_CHECK_ERROR(cuMemRelease(allocationHandle));  // after both cuMemExportToShareableHandle and cuMemMap
    }

    MappedDeviceMemory map() const &
    {
        return {allocationHandle, alignedAllocationSize, allocGranularity, memAllocationProp.location};
    }

    int exportToFd() const &
    {
        int fd = -1;
        CU_CHECK_ERROR(cuMemExportToShareableHandle(&fd, allocationHandle, kHandleType, 0));
        //::close(fd);
        return fd;
    }

private:
    const ::CUdevice cuDev;
    const size_t minAlignment;
    const size_t allocationSize;
    const size_t allocationAlignment;

    const ::CUmemAllocationProp memAllocationProp = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = kHandleType,
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = cuDev,
        },
        .win32HandleMetaData = nullptr,  // Win32 Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
        .allocFlags = {},
    };
    size_t allocGranularity = 0;
    size_t alignedAllocationSize = 0;
    ::CUmemGenericAllocationHandle allocationHandle = {};

    void init()
    {
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        SPDLOG_INFO("minimum allocGranularity {}", allocGranularity);
        alignedAllocationSize = utils::divUp(allocationSize, allocGranularity) * allocGranularity;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        SPDLOG_INFO("minAlignment {}, recommended allocGranularity {}", minAlignment, allocGranularity);
        allocGranularity = std::max({allocGranularity, allocationAlignment, minAlignment});
        CU_CHECK_ERROR(cuMemCreate(&allocationHandle, alignedAllocationSize, &memAllocationProp, 0));
    }
};

struct Tree::Impl : utils::OneTime<Impl>
{
    const Settings settings;
    const std::optional<DeviceUuidType> & deviceUuid;
    const size_t minAlignment;
    const scene_data::SceneData & sceneData;

    Impl(const Settings & settings, const std::optional<DeviceUuidType> & deviceUuid, size_t minAlignment, const scene_data::SceneData & sceneData)
        : settings{settings}
        , deviceUuid{deviceUuid}
        , minAlignment{minAlignment}
        , sceneData{sceneData}
    {}

    Impl(Impl &&) noexcept = default;

    bool build(const std::function<bool()> & cancel)
    {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        const CudaDevice cudaDevice{deviceUuid};
#endif
        auto triangles = sceneData.makeTriangles();
#if SAH_KD_TREE_HEADER_ONLY
        sah_kd_tree::Triangle<Traits> triangle{allocator};
#else
        sah_kd_tree::Triangle<Traits> triangle;
#endif
        triangle.setTriangle(triangles.begin(), triangles.end());
#if SAH_KD_TREE_HEADER_ONLY
        sah_kd_tree::Builder<Traits> builder{allocator};
        sah_kd_tree::Projection<Traits> x{allocator}, y{allocator}, z{allocator};
#else
        sah_kd_tree::Builder<Traits> builder;
        sah_kd_tree::Projection<Traits> x, y, z;
#endif
        sah_kd_tree::linkTriangles(triangle, x, y, z, builder);
        const sah_kd_tree::Params<Traits> params = {
            .emptinessFactor = settings.emptinessFactor,
            .traversalCost = settings.traversalCost,
            .intersectionCost = settings.intersectionCost,
            .maxDepth = settings.maxDepth,
        };
#if SAH_KD_TREE_HEADER_ONLY
        typename Traits::MemoryResource memoryResource;
        typename Traits::Allocator<void> allocator{&memoryResource};
        std::optional<sah_kd_tree::Tree<Traits>> tree{allocator};
#else
        std::optional<sah_kd_tree::Tree<Traits>> tree;
#endif
        tree = builder(cancel, params, x, y, z);
        if (!tree) {
            return false;
        }
        std::vector<Traits::U> layerDepth;
        layerDepth.reserve(std::size(tree->layerDepth));
        std::adjacent_difference(std::cbegin(tree->layerDepth), std::cend(tree->layerDepth), std::back_inserter(layerDepth));
        SPDLOG_INFO("Tree depth: {}", std::size(layerDepth));
        SPDLOG_INFO("Layer sizes: {}", layerDepth);
        SPDLOG_INFO("Polygon count: {}", std::size(tree->polygon.triangle));
        SPDLOG_INFO("Node count: {}", std::size(tree->node.parent));
        return true;
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const std::optional<DeviceUuidType> & deviceUuidType, size_t minAlignment, const scene_data::SceneData & sceneData)
    : impl_{std::make_unique<Impl>(settings, deviceUuidType, minAlignment, sceneData)}
{}

Tree::Tree(Tree &&) noexcept = default;
Tree::~Tree() = default;

auto Tree::getSettings() const & -> const Settings &
{
    return impl_->settings;
}

bool Tree::build(const std::function<bool()> & cancel)
{
    return impl_->build(cancel);
}

struct Builder::Impl : utils::OneTime<Impl>
{
    const Settings settings;

    Impl(const Settings & settings)
        : settings{settings}
    {
        printThrustVersion();
    }

    std::optional<Tree> build(const Tree::Settings & treeSettings, const scene_data::SceneData & sceneData, const std::function<bool()> & cancel) const
    {
        Tree tree{treeSettings, settings.deviceUuid, settings.minAlignment, sceneData};
        if (!tree.build(cancel)) {
            return {};
        }
        return tree;
    }

    static void printThrustVersion()
    {
        int major = THRUST_MAJOR_VERSION;
        int minor = THRUST_MINOR_VERSION;
        int subminor = THRUST_SUBMINOR_VERSION;
        int patch = THRUST_PATCH_NUMBER;
        SPDLOG_DEBUG("Thrust version: {}.{}.{}.{}", major, minor, subminor, patch);
        const char * deviceSystem = nullptr;
        switch (THRUST_DEVICE_SYSTEM) {
        case THRUST_DEVICE_SYSTEM_CUDA:
            deviceSystem = "CUDA";
            break;
        case THRUST_DEVICE_SYSTEM_OMP:
            deviceSystem = "OMP";
            break;
        case THRUST_DEVICE_SYSTEM_TBB:
            deviceSystem = "TBB";
            break;
        case THRUST_DEVICE_SYSTEM_CPP:
            deviceSystem = "CPP";
            break;
        }
        INVARIANT(deviceSystem, "{}", THRUST_DEVICE_SYSTEM);
        SPDLOG_DEBUG("Thrust device system: {}", deviceSystem);
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

Builder::Builder(const Settings & settings)
    : impl_{std::make_unique<Impl>(settings)}
{
    ASSERT((settings.minAlignment == 0) || std::has_single_bit(settings.minAlignment));
}

Builder::Builder(Builder &&) noexcept = default;
Builder::~Builder() = default;

std::optional<Tree> Builder::build(const Tree::Settings & settings, const scene_data::SceneData & sceneData, const std::function<bool()> & cancel) const
{
    return impl_->build(settings, sceneData, cancel);
}

}  // namespace builder
