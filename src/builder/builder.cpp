#include <builder/builder.hpp>
#include <utils/assert.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <utils/assert.hpp>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_allocator.h>
#include <thrust/system/cuda/pointer.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <utils/math.hpp>
#include <utils/auto_cast.hpp>
#include <scene_data/scene_data.hpp>

#include <algorithm>
#include <iterator>

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
struct fmt::formatter<CUresult> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(CUresult result, FormatContext & ctx) const
    {
        const char * errorName = "unknown";
        const char * errorString = "unknown";
        ::cuGetErrorName(result, &errorName);
        ::cuGetErrorString(result, &errorString);
        return fmt::format_to(ctx.out(), "{}: {}", errorName, errorString);
    }
};

namespace sah_kd_tree_fd
{

namespace
{

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#error "Only Thrust device system CUDA is currently supported"
#endif
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

}

struct Tree::Impl : utils::OneTime<Impl>
{
    // Win32 CU_MEM_HANDLE_TYPE_WIN32
    static constexpr CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    const Settings settings;
    const scene_data::SceneData & sceneData;

    CUdevice cuDev = {};
    sah_kd_tree::Tree tree;

    Impl(const Settings & settings, const scene_data::SceneData & sceneData)
        : settings{settings}
        , sceneData{sceneData}
    {
        printThrustVersion();
        settings.check();
    }

    Impl(Impl &&) noexcept = default;

    static void printThrustVersion()
    {
        int major    = THRUST_MAJOR_VERSION;
        int minor    = THRUST_MINOR_VERSION;
        int subminor = THRUST_SUBMINOR_VERSION;
        int patch    = THRUST_PATCH_NUMBER;
        SPDLOG_DEBUG("Thrust version: {}.{}.{}.{}", major, minor, subminor, patch);
    }

    void selectDevice()
    {
        int devCount = 0;
        CUDA_CHECK_ERROR(cudaGetDeviceCount(&devCount));
        INVARIANT(devCount > 0, "");
        {
            int cuDevCount = 0;
            CU_CHECK_ERROR(cuDeviceGetCount(&cuDevCount));
            ASSERT(devCount == cuDevCount);
        }

        ASSERT(std::size(settings.deviceUuid) == sizeof cudaDeviceProp::uuid);
        ASSERT(std::size(settings.deviceUuid) == sizeof(CUuuid));
        int cudaDev = cudaInvalidDeviceId;
        for (int i = 0; i < devCount; ++i) {
            cudaDeviceProp devProp = {};
            CUDA_CHECK_ERROR(cudaGetDeviceProperties(&devProp, i));
            if (std::memcmp(&devProp.uuid, std::data(settings.deviceUuid), sizeof devProp.uuid) == 0) {
                cudaDev = i;
                {
                    CU_CHECK_ERROR(cuDeviceGet(&cuDev, i));
                    CUuuid uuid = {};
                    CU_CHECK_ERROR(cuDeviceGetUuid_v2(&uuid, cuDev));
                    INVARIANT(std::memcmp(&uuid, std::data(settings.deviceUuid), sizeof uuid), "");
                }
                break;
            }
        }
        INVARIANT(cudaDev != cudaInvalidDeviceId, "No matching by UUID devices found");


        int devComputeModeSupported = 0;
        CU_CHECK_ERROR(cuDeviceGetAttribute(
            &devComputeModeSupported, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev));
        INVARIANT(devComputeModeSupported != CU_COMPUTEMODE_DEFAULT, "{}", devComputeModeSupported);

        int virtualAddressManagementSupported = 0;
        CU_CHECK_ERROR(cuDeviceGetAttribute(&virtualAddressManagementSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDev));
        INVARIANT(virtualAddressManagementSupported != 0, "Virtual address management is not supported");

        int fdSupported = 0;
        // Win32: CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
        CU_CHECK_ERROR(cuDeviceGetAttribute(&fdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cuDev));
        INVARIANT(fdSupported != 0, "Posix file descriptor handle type is not supported");

        CUDA_CHECK_ERROR(cudaSetDevice(cudaDev));
    }

    void test(size_t allocationSize, size_t allocationAlignment)
    {
        CUmemAllocationProp memAllocationProp = {
                                                 .type = CU_MEM_ALLOCATION_TYPE_PINNED,
                                                 .requestedHandleTypes = kHandleType,
                                                 .location = {
                                                     .type = CU_MEM_LOCATION_TYPE_DEVICE,
                                                     .id = cuDev,
                                                 },
                                                 .win32HandleMetaData = nullptr,  // Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp
                                                 .allocFlags = {},
                                                 };
        size_t allocGranularity = 0;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        SPDLOG_INFO("Minimum allocGranularity {}", allocGranularity);
        allocationSize = utils::divUp(allocationSize, allocGranularity) * allocationSize;
        CU_CHECK_ERROR(cuMemGetAllocationGranularity(&allocGranularity, &memAllocationProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        SPDLOG_INFO("minAlignment {}, allocGranularity {}", settings.minAlignment, allocGranularity);
        allocGranularity = std::max({allocGranularity, allocationAlignment, settings.minAlignment});

        CUmemGenericAllocationHandle allocationHandle = {};
        CU_CHECK_ERROR(cuMemCreate(&allocationHandle, allocationSize, &memAllocationProp, 0));
        {
            CUdeviceptr devPtr = {};
            CU_CHECK_ERROR(cuMemAddressReserve(&devPtr, allocationSize, allocGranularity, devPtr, 0));
            CU_CHECK_ERROR(cuMemMap(devPtr, allocationSize, 0, allocationHandle, 0));
            {
                CUmemAccessDesc accessDescriptor[] = {
                    {
                        .location = memAllocationProp.location,
                        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                    },
                };
                CU_CHECK_ERROR(cuMemSetAccess(devPtr, allocationSize, std::data(accessDescriptor), std::size(accessDescriptor)));
                void * p = utils::autoCast(devPtr);
            }
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

    void build()
    {
        selectDevice();
        test(1, 0);

        VulkanMemoryResource vmr;
        using T = int;
        {
            using Allocator = thrust::mr::allocator<void, MemoryResourceBase>;
            Allocator allocator{&vmr};
            thrust::device_vector<T, Allocator::template rebind<T>::other> v{allocator};
        }
        {
            thrust::device_ptr_memory_resource<MemoryResourceBase> mr{&vmr};
            thrust::mr::polymorphic_adaptor_resource<thrust::device_ptr<void>> adaptor{&mr};
            using Allocator = thrust::mr::polymorphic_allocator<T, thrust::device_ptr<void>>;
            Allocator allocator{&adaptor};
            thrust::device_vector<T, Allocator> v{allocator};
        }

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        auto triangles = sceneData.makeTriangles();
        sah_kd_tree::Triangle triangle;
        triangle.setTriangle(triangles.begin(), triangles.end());

        sah_kd_tree::Builder builder;
        sah_kd_tree::Projection x, y, z;
        sah_kd_tree::linkTriangles(triangle, x, y, z, builder);
        sah_kd_tree::Params params = {
            .emptinessFactor = settings.emptinessFactor,
            .traversalCost = settings.traversalCost,
            .intersectionCost = settings.intersectionCost,
            .maxDepth = settings.maxDepth,
        };
        tree = builder(params, x, y, z);
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const scene_data::SceneData & sceneData)
    : impl_{std::make_shared<Impl>(settings, sceneData)}
{}

Tree::Tree(Tree &&) noexcept = default;
Tree::~Tree() = default;

void Tree::build()
{
    return impl_->build();
}

}  // namespace sah_kd_tree_fd
