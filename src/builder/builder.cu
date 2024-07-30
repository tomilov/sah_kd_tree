#include <builder/builder.hpp>
#include <utils/assert.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_allocator.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/mr/device_memory_resource.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <utils/math.hpp>
#include <utils/auto_cast.hpp>
#include <scene_data/scene_data.hpp>
#include <fmt/std.h>

#include <bit>
#include <algorithm>
#include <iterator>
#include <utility>

#include <cstddef>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>

#define CUDA_CHECK_ERROR(call) do { cudaError error = cudaSuccess; INVARIANT((error = (call)) == cudaSuccess, "{}", error); } while (false)
#define CU_CHECK_ERROR(call) do { CUresult result = CUDA_SUCCESS; INVARIANT((result = (call)) == CUDA_SUCCESS, "{}", result); } while (false)

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

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using MemoryResourceBase = thrust::mr::memory_resource<thrust::cuda::pointer<void>>;

class VulkanMemoryResource final : public MemoryResourceBase
{
public:
    VulkanMemoryResource() = default;

    pointer do_allocate(size_t bytes, size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        (void)bytes;
        (void)alignment;
        pointer ret = nullptr;
        return ret;
    }

    void do_deallocate(pointer p, size_t bytes, size_t alignment) override
    {
        (void)p;
        (void)bytes;
        (void)alignment;
    }
};
#endif

#if SAH_KD_TREE_HEADER_ONLY
struct Traits : sah_kd_tree::DefaultTraits  // cannot be member typedef of Tree::Impl because of wierd CUDA parser
{
    using DefaultTraits::I;
    using DefaultTraits::U;
    using DefaultTraits::F;
    using MemoryResource = thrust::device_memory_resource;
    using Allocator = thrust::mr::allocator<void, MemoryResource>;
};
#else
using Traits = sah_kd_tree::DefaultTraits;
#endif

}

class CudaDevice : utils::OneTime<CudaDevice>
{
public:
    CudaDevice(bool skipDeviceCheck, const Builder::Settings::DeviceUuidType & deviceUuid)
        : skipDeviceCheck{skipDeviceCheck}
        , deviceUuid{deviceUuid}
    {
        checkTraits();
        selectCudaDevice();
    }

    CudaDevice(CudaDevice && rhs) noexcept
        : skipDeviceCheck{rhs.skipDeviceCheck}
        , deviceUuid{rhs.deviceUuid}
    {
        std::swap(cudaDev, rhs.cudaDev);
        std::swap(cuDev, rhs.cuDev);
    }

    const ::CUdevice & getCuDevice() const &
    {
        return cuDev;
    }

private:
    const bool skipDeviceCheck;
    const Builder::Settings::DeviceUuidType deviceUuid;

    int cudaDev = cudaInvalidDeviceId;
    ::CUdevice cuDev = CU_DEVICE_INVALID;

    void selectCudaDevice()
    {
        cudaDeviceProp devProp = {};
        ASSERT(std::size(deviceUuid) == sizeof cudaDeviceProp::uuid);
        CUDA_CHECK_ERROR(cudaGetDevice(&cudaDev));
        if (cudaDev != cudaInvalidDeviceId) {
            CUDA_CHECK_ERROR(cudaGetDeviceProperties(&devProp, cudaDev));
            selectCuDevice(devProp.uuid);
        } else {
            int devCount = 0;
            CUDA_CHECK_ERROR(cudaGetDeviceCount(&devCount));
            INVARIANT(devCount > 0, "");
            for (cudaDev = 0; cudaDev < devCount; ++cudaDev) {
                if (skipDeviceCheck || (std::memcmp(&devProp.uuid, std::data(deviceUuid), sizeof devProp.uuid) == 0)) {
                    break;
                }
            }
            INVARIANT(cudaDev != devCount, "No matching by UUID devices found using CUDA Runtime API");
            selectCuDevice(devProp.uuid);
            CUDA_CHECK_ERROR(cudaInitDevice(cudaDev, 0, 0));
            CUDA_CHECK_ERROR(cudaSetDevice(cudaDev));
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
                if (std::memcmp(&cudaDeviceUuid, &uuid, sizeof(uuid)) == 0) {
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

struct Tree::Impl : utils::OneTime<Impl>
{
    const Settings settings;
    const CudaDevice & cudaDevice;
    const scene_data::SceneData & sceneData;

#if SAH_KD_TREE_HEADER_ONLY
    typename Traits::MemoryResource memoryResource;
    typename Traits::Allocator allocator{&memoryResource};
    sah_kd_tree::Tree<Traits> tree{allocator};
#else
    sah_kd_tree::Tree<Traits> tree;
#endif

    Impl(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneData & sceneData)
        : settings{settings}
        , cudaDevice{cudaDevice}
        , sceneData{sceneData}
    {}

    Impl(Impl &&) noexcept = default;

    bool build()
    {
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
        tree = builder(params, x, y, z);
        return true;
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneData & sceneData)
    : impl_{std::make_shared<Impl>(settings, cudaDevice, sceneData)}
{}

Tree::Tree(Tree &&) noexcept = default;
Tree::~Tree() = default;

auto Tree::getSettings() const & -> const Settings &
{
    return impl_->settings;
}

bool Tree::build()
{
    return impl_->build();
}

struct Builder::Impl : utils::OneTime<Impl>
{
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr ::CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    const Settings settings;
    CudaDevice cudaDevice;

    Impl(const Settings & settings)
        : settings{settings}
        , cudaDevice{settings.skipDeviceCheck, settings.deviceUuid}
    {
        printThrustVersion();
    }

    std::optional<Tree> build(const Tree::Settings & treeSettings, const scene_data::SceneData & sceneData) const
    {
        //test(1, 0);

        Tree tree{treeSettings, cudaDevice, sceneData};
        try {
            if (!tree.build()) {
                return {};
            }
        } catch (const std::bad_alloc & e) {
            SPDLOG_ERROR("{}", e);
            return {};
        }
        return tree;
    }

    void test(size_t allocationSize, size_t allocationAlignment) const
    {
        const ::CUmemAllocationProp memAllocationProp = {
            .type = CU_MEM_ALLOCATION_TYPE_PINNED,
            .requestedHandleTypes = kHandleType,
            .location = {
                .type = CU_MEM_LOCATION_TYPE_DEVICE,
                .id = cudaDevice.getCuDevice(),
            },
            .win32HandleMetaData = nullptr,  // Win32 Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
            .allocFlags = {},
        };
        size_t allocGranularity = 0;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        SPDLOG_INFO("minimum allocGranularity {}", allocGranularity);
        allocationSize = utils::divUp(allocationSize, allocGranularity) * allocGranularity;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        SPDLOG_INFO("minAlignment {}, recommended allocGranularity {}", settings.minAlignment, allocGranularity);
        allocGranularity = std::max({allocGranularity, allocationAlignment, settings.minAlignment});

        ::CUmemGenericAllocationHandle allocationHandle = {};
        CU_CHECK_ERROR(cuMemCreate(&allocationHandle, allocationSize, &memAllocationProp, 0));
        {
            ::CUdeviceptr devPtr = {};
            CU_CHECK_ERROR(cuMemAddressReserve(&devPtr, allocationSize, allocGranularity, devPtr, 0));
            CU_CHECK_ERROR(cuMemMap(devPtr, allocationSize, 0, allocationHandle, 0));
            {
                ::CUmemAccessDesc accessDescriptor[] = {
                    {
                        .location = memAllocationProp.location,
                        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                    },
                };
                CU_CHECK_ERROR(cuMemSetAccess(devPtr, allocationSize, std::data(accessDescriptor), std::size(accessDescriptor)));
                void * p = utils::autoCast(devPtr);
            }
            CU_CHECK_ERROR(cuMemUnmap(devPtr, allocationSize));
            CU_CHECK_ERROR(cuMemAddressFree(devPtr, allocationSize));
        }
        {
            int fd = -1;
            CU_CHECK_ERROR(cuMemExportToShareableHandle(&fd, allocationHandle, kHandleType, 0));
            //
            ::close(fd);
        }
        CU_CHECK_ERROR(cuMemRelease(allocationHandle)); // after both cuMemExportToShareableHandle and cuMemMap
    }

    static void printThrustVersion()
    {
        int major = THRUST_MAJOR_VERSION;
        int minor = THRUST_MINOR_VERSION;
        int subminor = THRUST_SUBMINOR_VERSION;
        int patch = THRUST_PATCH_NUMBER;
        SPDLOG_DEBUG("Thrust version: {}.{}.{}.{}", major, minor, subminor, patch);
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

Builder::Builder(const Settings & settings)
    : impl_{std::make_shared<Impl>(settings)}
{
    ASSERT((settings.minAlignment == 0) || std::has_single_bit(settings.minAlignment));
}

Builder::Builder(Builder &&) noexcept = default;
Builder::~Builder() = default;

std::optional<Tree> Builder::build(const Tree::Settings & settings, const scene_data::SceneData & sceneData) const
{
    return impl_->build(settings, sceneData);
}

}  // namespace builder
