#include <builder/builder.cuh>
#include <compute/compute.hpp>
#include <sah_kd_tree/sah_kd_tree_inline.cuh>

#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/vector.h>

namespace builder
{

struct ThrustDeviceSystemCUDA
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    using MemoryResource = thrust::cuda::memory_resource;
    template<typename T>
    using Allocator = thrust::mr::allocator<T, MemoryResource>;
    template<typename T>
    using Vector = thrust::cuda::vector<T, Allocator<T>>;

    struct TreeContext
    {
        MemoryResource memoryResource;
        const Allocator<std::byte> allocator{&memoryResource};

        sah_kd_tree::Tree<ThrustDeviceSystemCUDA> tree{allocator};
    };

    using Exec = thrust::cuda_cub::par_nosync_t::execute_with_allocator_type<Allocator<std::byte>>::type;
    using Progress = sah_kd_tree::DefaultTraits::Progress;

    struct BuildContext
    {
        const Allocator<std::byte> allocator;
        const compute::CudaStream cudaStream;
        const Exec exec;

        sah_kd_tree::Builder<ThrustDeviceSystemCUDA> builder{allocator, exec};
        sah_kd_tree::Projection<ThrustDeviceSystemCUDA> x{allocator, exec}, y{allocator, exec}, z{allocator, exec};

        sah_kd_tree::Triangle<ThrustDeviceSystemCUDA> triangle{allocator, exec};

        explicit BuildContext(const TreeContext & treeContext)
            : allocator{treeContext.allocator}
            , exec{thrust::cuda::par_nosync(allocator).on(cudaStream.getHandle())}
        {}

        ~BuildContext()
        {
            cudaStream.synchronize();
        }
    };
};

template class TreeBuildContext<ThrustDeviceSystemCUDA>;
template TreePtr build<ThrustDeviceSystemCUDA>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
