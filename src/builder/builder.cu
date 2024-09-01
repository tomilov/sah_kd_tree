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
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <cstddef>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

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

class OutOfMemoryException : public std::bad_alloc
{
    const char * what() const noexcept override
    {
        return "OutOfMemoryException: out of memory";
    }
};

#if SAH_KD_TREE_HEADER_ONLY
struct Traits  // cannot be member typedef of Tree::Impl because of wierd CUDA parser
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    using MemoryResource = thrust::device_memory_resource;
    template<typename T>
    using Allocator = thrust::mr::allocator<T, MemoryResource>;
    template<typename T>
    using Vector = thrust::device_vector<T, Allocator<T>>;
    using Progress = std::function<bool(size_t progressValue)>;
};
#else
using Traits = sah_kd_tree::DefaultTraits;
#endif

class DeviceMemory;

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

    ::CUdeviceptr getPtr() const
    {
        return devPtr;
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

class DeviceMemory : utils::OneTime<DeviceMemory>
{
public:
    DeviceMemory(::CUdevice cuDev, size_t allocationSize, size_t allocationAlignment = 0)
        : memAllocationProp{makeMemAllocationProp(cuDev)}
        , allocGranularity{getAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)}
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

    utils::Fd exportMemoryObject() const
    {
        int fd = -1;
        CU_CHECK_ERROR(cuMemExportToShareableHandle(&fd, allocationHandle, kHandleType, 0));
        return utils::Fd{fd};
    }

    size_t getSize() const
    {
        return alignedAllocationSize;
    }

private:
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

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
                .id = cuDev,
            },
            .win32HandleMetaData = nullptr,  // Win32 Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
            .allocFlags = {},
        };
    }

    size_t getAllocationGranularity(CUmemAllocationGranularity_flags_enum memAllocationGranularityFlag) const
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
        INVARIANT(kind, "{}", fmt::underlying(memAllocationGranularityFlag));
        SPDLOG_INFO("{} allocGranularity {}", kind, allocGranularity);
        return allocGranularity;
    }

    size_t getAlignedAllocationSize(size_t allocationSize, size_t allocationAlignment) const
    {
        const size_t recommendedAllocGranularity = getAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        return utils::divUp(std::max(allocationSize, allocationAlignment), recommendedAllocGranularity) * recommendedAllocGranularity;
    }

    ::CUmemGenericAllocationHandle makeMemGenericAllocationHandle() const
    {
        ::CUmemGenericAllocationHandle allocationHandle = {};
        const auto result = cuMemCreate(&allocationHandle, alignedAllocationSize, &memAllocationProp, 0);
        if (result == CUDA_ERROR_OUT_OF_MEMORY) {
            throw OutOfMemoryException{};
        }
        INVARIANT(result == CUDA_SUCCESS, "{}", result);
        return allocationHandle;
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace

class CudaDevice
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
    const std::optional<DeviceUuidType> deviceUuid;

    int cudaDev = cudaInvalidDeviceId;
    ::CUdevice cuDev = CU_DEVICE_INVALID;

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

    void selectCudaDevice()
    {
        cudaDeviceProp devProp = {};
        static_assert(sizeof(DeviceUuidType) == sizeof devProp.uuid);
        {
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
        }
        selectCuDevice(devProp.uuid);
        CUDA_CHECK_ERROR(cudaSetDevice(cudaDev));
    }
};

struct Tree::Impl : utils::OneTime<Impl>
{
    const Settings settings;
    const scene_data::SceneDataWeakPtr sceneData;

    size_t dataSize = 0;
    size_t allocationSize = 0;
    size_t triangleCount = 0;
    std::vector<size_t> layerSizes;
    size_t polygonCount = 0;
    size_t nodeCount = 0;
    std::optional<utils::Fd> fd;

    Impl(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
        : settings{settings}
        , sceneData{sceneData}
    {
        CUDA_CHECK_ERROR(cudaSetDevice(cudaDevice.getCudaDev()));
        ASSERT(sceneData);
        auto triangles = sceneData->makeTriangles();
#if SAH_KD_TREE_HEADER_ONLY
        typename Traits::MemoryResource memoryResource;
        typename Traits::Allocator<void> allocator{&memoryResource};
        sah_kd_tree::Triangle<Traits> triangle{allocator};
#else
        sah_kd_tree::Triangle<Traits> triangle;
#endif
        triangle.setTriangle(triangles.begin(), triangles.end());
#if SAH_KD_TREE_HEADER_ONLY
        sah_kd_tree::Builder<Traits> builder{allocator};
        sah_kd_tree::Projection<Traits> x{allocator}, y{allocator}, z{allocator};
        sah_kd_tree::Tree<Traits> tree{allocator};
#else
        sah_kd_tree::Builder<Traits> builder;
        sah_kd_tree::Projection<Traits> x, y, z;
        sah_kd_tree::Tree<Traits> tree;
#endif
        sah_kd_tree::linkTriangles(triangle, x, y, z, builder);
        const sah_kd_tree::Params<Traits> params = {
            .emptinessFactor = settings.emptinessFactor,
            .traversalCost = settings.traversalCost,
            .intersectionCost = settings.intersectionCost,
            .maxDepth = settings.maxDepth,
        };
        if (!builder.build(progress, params, x, y, z, tree)) {
            return;
        }
        static_assert(std::is_same_v<Traits::F, glm::float32>);
        static_assert(std::is_same_v<Traits::U, glm::uint32>);
        static_assert(std::is_same_v<Traits::I, glm::int32>);
        triangleCount = triangles.getCount();
        populateTreeSizes(tree);
        const size_t trianglesSize = triangleCount * sizeof(scene_data::Triangle);
        dataSize += trianglesSize;
        const auto gatherSize = [this]<typename Vector>(const Vector & v)
        {
            dataSize += std::size(v) * sizeof(typename Vector::value_type);
        };
        traverseTree(tree, gatherSize);
        SPDLOG_INFO("Allocation size for tree: {}", dataSize);
        DeviceMemory deviceMemory{cudaDevice.getCuDev(), dataSize};
        allocationSize = deviceMemory.getSize();
        SPDLOG_INFO("Tree size {}, allocation size {}", dataSize, allocationSize);
        {
            auto mappedDeviceMemory = deviceMemory.map();
            const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();
            ::CUdeviceptr p = devPtr;
            {
                CU_CHECK_ERROR(::cuMemcpyHtoD(p, triangles.begin(), trianglesSize));
                p += trianglesSize;
            }
            const auto gatherData = [&p]<typename Vector>(const Vector & v)
            {
                const ::CUdeviceptr src = utils::autoCast(thrust::raw_pointer_cast(std::data(v)));
                const size_t size = std::size(v) * sizeof(typename Vector::value_type);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
                CU_CHECK_ERROR(::cuMemcpyDtoD(p, src, size));
#else
                CU_CHECK_ERROR(::cuMemcpyHtoD(p, src, size));
#endif
                p += size;
            };
            traverseTree(tree, gatherData);
            ASSERT_MSG(p == devPtr + dataSize, "{} ^ {}", p, devPtr + dataSize);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        }
        fd.emplace(deviceMemory.exportMemoryObject());
    }

    Impl(Impl &&) noexcept = default;

    void populateTreeSizes(const sah_kd_tree::Tree<Traits> & tree)
    {
        ASSERT(std::is_sorted(std::cbegin(tree.layerDepth), std::cend(tree.layerDepth)));
        layerSizes.resize(std::size(tree.layerDepth));
        std::adjacent_difference(std::cbegin(tree.layerDepth), std::cend(tree.layerDepth), std::begin(layerSizes));
        polygonCount = std::size(tree.polygon.triangle);
        nodeCount = std::size(tree.node.parent);
        SPDLOG_INFO("Tree depth: {}", std::size(layerSizes));
        SPDLOG_INFO("Layer sizes: {}", layerSizes);
        SPDLOG_INFO("Polygon count: {}", polygonCount);
        SPDLOG_INFO("Node count: {}", nodeCount);
    }

    template<typename F>
    static void traverseProjection(const sah_kd_tree::Tree<Traits>::Projection & projection, const F & f)
    {
        f(projection.node.min);
        f(projection.node.max);
        f(projection.node.leftRope);
        f(projection.node.rightRope);
    };

    template<typename F>
    static void traverseTree(const sah_kd_tree::Tree<Traits> & tree, const F & f)
    {
        traverseProjection(tree.x, f);
        traverseProjection(tree.y, f);
        traverseProjection(tree.z, f);
        f(tree.polygon.triangle);
        f(tree.node.splitDimension);
        f(tree.node.splitPos);
        f(tree.node.leftChild);
        f(tree.node.rightChild);
        f(tree.node.parent);
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
    : impl_{std::make_unique<Impl>(settings, cudaDevice, sceneData, progress)}
{}

Tree::Tree(Tree &&) noexcept = default;
Tree::~Tree() = default;

auto Tree::getSettings() const & -> const Settings &
{
    return impl_->settings;
}

scene_data::SceneDataPtr Tree::getSceneData() const
{
    return impl_->sceneData.lock();
}

bool Tree::isEmpty() const
{
    return !impl_->fd;
}

utils::Fd Tree::getFd() &&
{
    ASSERT(!isEmpty());
    utils::Fd fd = std::move(impl_->fd).value();
    impl_->fd.reset();
    return fd;
}

utils::Fd Tree::cloneFd() const &
{
    ASSERT(!isEmpty());
    return impl_->fd.value().clone();
}

size_t Tree::getDataSize() const
{
    ASSERT(impl_->dataSize > 0);
    return impl_->dataSize;
}

size_t Tree::getAllocationSize() const
{
    ASSERT(impl_->allocationSize > 0);
    return impl_->allocationSize;
}

size_t Tree::getTriangleCount() const
{
    ASSERT(impl_->triangleCount > 0);
    return impl_->triangleCount;
}

const std::vector<size_t> & Tree::getLayerSizes() const &
{
    ASSERT(!std::empty(impl_->layerSizes));
    return impl_->layerSizes;
}

size_t Tree::getPolygonCount() const
{
    ASSERT(impl_->polygonCount > 0);
    return impl_->polygonCount;
}

size_t Tree::getNodeCount() const
{
    ASSERT(impl_->nodeCount > 0);
    return impl_->nodeCount;
}

struct Builder::Impl : utils::OneTime<Impl>
{
    CudaDevice cudaDevice;

    Impl(const std::optional<DeviceUuidType> & deviceUuid)
        : cudaDevice{deviceUuid}
    {
        printThrustVersion();
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

Builder::Builder(const std::optional<DeviceUuidType> & deviceUuid)
    : impl_{std::make_unique<Impl>(deviceUuid)}
{}

Builder::Builder(Builder &&) noexcept = default;
Builder::~Builder() = default;

std::optional<Tree> Builder::build(const Tree::Settings & treeSettings, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) const
{
    Tree tree{treeSettings, impl_->cudaDevice, sceneData, progress};
    if (tree.isEmpty()) {
        return std::nullopt;
    }
    return tree;
}

}  // namespace builder
