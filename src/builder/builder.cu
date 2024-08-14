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
#include <type_traits>

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

class CudaDevice : utils::OneTime<CudaDevice>
{
public:
    CudaDevice(const std::optional<DeviceUuidType> & deviceUuid)
        : deviceUuid{deviceUuid}
    {
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

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class DeviceMemory : utils::OneTime<DeviceMemory>
{
public:
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    class MappedDeviceMemory : utils::OneTime<MappedDeviceMemory>
    {
    public:
        MappedDeviceMemory(const ::CUmemLocation & location, size_t allocGranularity, size_t alignedAllocationSize, ::CUmemGenericAllocationHandle allocationHandle)
            : alignedAllocationSize{alignedAllocationSize}
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

        MappedDeviceMemory(MappedDeviceMemory && rhs) noexcept
            : alignedAllocationSize{rhs.alignedAllocationSize}
            , devPtr{std::exchange(rhs.devPtr, devPtr)}
        {}

        ~MappedDeviceMemory()
        {
            if (devPtr == ::CUdeviceptr{}) {
                return;
            }
            CU_CHECK_ERROR(cuMemUnmap(devPtr, alignedAllocationSize));
            CU_CHECK_ERROR(cuMemAddressFree(devPtr, alignedAllocationSize));
        }

        void * getPtr() const
        {
            return utils::autoCast(devPtr);
        }

    private:
        friend DeviceMemory;

        const size_t alignedAllocationSize;

        ::CUdeviceptr devPtr = {};

        static constexpr void completeClassContext [[maybe_unused]] ()
        {
            checkTraits();
        }
    };

    DeviceMemory(::CUdevice cuDev, size_t allocationSize, size_t allocationAlignment = 0)
        : memAllocationProp{makeMemAllocationProp(cuDev)}
        , allocGranularity{getAllocMinGranularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)}
        , alignedAllocationSize{getAlignedAllocationSize(allocationSize, allocationAlignment)}
        , allocationHandle{makeMemGenericAllocationHandle()}
    {}

    DeviceMemory(DeviceMemory && rhs) noexcept
        : memAllocationProp{rhs.memAllocationProp}
        , allocGranularity{rhs.allocGranularity}
        , alignedAllocationSize{rhs.alignedAllocationSize}
        , allocationHandle{std::exchange(rhs.allocationHandle, allocationHandle)}
    {}

    ~DeviceMemory()
    {
        if (allocationHandle == ::CUmemGenericAllocationHandle{}) {
            return;
        }
        CU_CHECK_ERROR(cuMemRelease(allocationHandle));  // after both cuMemExportToShareableHandle and cuMemMap
    }

    MappedDeviceMemory map() const
    {
        return {memAllocationProp.location, allocGranularity, alignedAllocationSize, allocationHandle};
    }

    File exportMemoryObject() const
    {
        int fd = -1;
        CU_CHECK_ERROR(cuMemExportToShareableHandle(&fd, allocationHandle, kHandleType, 0));
        INVARIANT(fd >= 0, "");
        return File::make(fd);
    }

private:
    const ::CUmemAllocationProp memAllocationProp;
    const size_t allocGranularity;
    const size_t alignedAllocationSize = 0;
    ::CUmemGenericAllocationHandle allocationHandle = {};

    static ::CUmemAllocationProp makeMemAllocationProp(::CUdevice cuDev)
    {
        return {
            .type = CU_MEM_ALLOCATION_TYPE_PINNED,
            .requestedHandleTypes = kHandleType,
            .location = {
                .type = CU_MEM_LOCATION_TYPE_DEVICE,
            },
            .win32HandleMetaData = nullptr,  // Win32 Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
            .allocFlags = {},
        };
    }

    size_t getAllocMinGranularity(CUmemAllocationGranularity_flags_enum memAllocationGranularityFlag) const
    {
        size_t allocGranularity = 0;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, memAllocationGranularityFlag));
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
        SPDLOG_INFO("{} allocGranularity {}", kind, allocGranularity);
        return allocGranularity;
    }

    size_t getAlignedAllocationSize(size_t allocationSize, size_t allocationAlignment) const
    {
        const size_t recommendedAllocGranularity = getAllocMinGranularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        const size_t alignment = std::max(recommendedAllocGranularity, allocationAlignment);
        return utils::divUp(allocationSize, alignment) * alignment;
    }

    ::CUmemGenericAllocationHandle makeMemGenericAllocationHandle() const
    {
        ::CUmemGenericAllocationHandle allocationHandle = {};
        CU_CHECK_ERROR(cuMemCreate(&allocationHandle, alignedAllocationSize, &memAllocationProp, 0));
        // TODO: CUDA_ERROR_OUT_OF_MEMORY -> std::bad_alloc
        return allocationHandle;
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace

File::File(File && file) noexcept
    : fd{std::exchange(file.fd, fd)}
{
    INVARIANT(fd >= 0, "");
}

File::~File()
{
    if (fd < 0) {
        return;
    }
    ::close(fd);
}

File File::make(int fd)
{
    return File{fd};
}

File File::dup(int fd)
{
    return File{::dup(fd)};
}

File::File(int fd)
    : fd{fd}
{}

struct Tree::Impl : utils::OneTime<Impl>
{
    const Settings settings;
    const std::optional<DeviceUuidType> & deviceUuid;
    const scene_data::SceneData & sceneData;

    Impl(const Settings & settings, const std::optional<DeviceUuidType> & deviceUuid, const scene_data::SceneData & sceneData)
        : settings{settings}
        , deviceUuid{deviceUuid}
        , sceneData{sceneData}
    {}

    Impl(Impl &&) noexcept = default;

    static size_t getTreeSize(const sah_kd_tree::Tree<Traits> & tree)
    {
        std::vector<Traits::U> layerDepth;
        layerDepth.reserve(std::size(tree.layerDepth));
        std::adjacent_difference(std::cbegin(tree.layerDepth), std::cend(tree.layerDepth), std::back_inserter(layerDepth));
        SPDLOG_INFO("Tree depth: {}", std::size(layerDepth));
        SPDLOG_INFO("Layer sizes: {}", layerDepth);
        SPDLOG_INFO("Polygon count: {}", std::size(tree.polygon.triangle));
        SPDLOG_INFO("Node count: {}", std::size(tree.node.parent));
        size_t allocationSize = 0;
        static constexpr auto getVectorSize = []<typename Vector>(const Vector & v) -> size_t
        {
            return std::size(v) * sizeof(typename Vector::value_type);
        };
        static constexpr auto getProjectionSize = [](const auto & p) -> size_t
        {
            return getVectorSize(p.node.min) + getVectorSize(p.node.max) + getVectorSize(p.node.leftRope) + getVectorSize(p.node.rightRope);
        };
        allocationSize += getProjectionSize(tree.x) + getProjectionSize(tree.y) + getProjectionSize(tree.z);
        allocationSize += getVectorSize(tree.polygon.triangle);
        allocationSize += getVectorSize(tree.node.splitDimension) + getVectorSize(tree.node.splitPos) + getVectorSize(tree.node.leftChild) + getVectorSize(tree.node.rightChild) + getVectorSize(tree.node.parent);
        SPDLOG_INFO("Allocation size for tree: {}", allocationSize);
        return allocationSize;
    }

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
        static_assert(std::is_same_v<Traits::F, glm::float32>);
        static_assert(std::is_same_v<Traits::U, glm::uint32>);
        static_assert(std::is_same_v<Traits::I, glm::int32>);
        size_t allocationSize = getTreeSize(tree.value());
        DeviceMemory deviceMemory{cudaDevice.getCudaDev(), allocationSize};
        auto mappedDeviceMemory = deviceMemory.map();
        return true;
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const std::optional<DeviceUuidType> & deviceUuidType, const scene_data::SceneData & sceneData)
    : impl_{std::make_unique<Impl>(settings, deviceUuidType, sceneData)}
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
        Tree tree{treeSettings, settings.deviceUuid, sceneData};
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

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Builder::Builder(const Settings & settings)
    : impl_{std::make_unique<Impl>(settings)}
{}

Builder::Builder(Builder &&) noexcept = default;
Builder::~Builder() = default;

std::optional<Tree> Builder::build(const Tree::Settings & settings, const scene_data::SceneData & sceneData, const std::function<bool()> & cancel) const
{
    return impl_->build(settings, sceneData, cancel);
}

}  // namespace builder
