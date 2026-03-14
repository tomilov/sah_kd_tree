#include <builder/builder.cuh>
#include <compute/compute.hpp>
#include <sah_kd_tree/sah_kd_tree_inline.cuh>

#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/vector.h>

namespace builder
{

template<>
struct BuilderContext<ThrustDeviceSystem::CUDA>
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    using MemoryResource = thrust::cuda::memory_resource;
    template<typename T>
    using Allocator = thrust::mr::allocator<T, MemoryResource>;
    template<typename T>
    using Vector = thrust::cuda::vector<T, Allocator<T>>;
    using ComponentIterator = typename Vector<F>::const_pointer;

    struct TreeContext
    {
        MemoryResource memoryResource;
        const Allocator<std::byte> allocator{&memoryResource};

        sah_kd_tree::Tree<BuilderContext> tree{allocator};
    };

    using Exec = thrust::cuda_cub::par_nosync_t::execute_with_allocator_type<Allocator<std::byte>>::type;
    using Progress = sah_kd_tree::DefaultTraits::Progress;

    struct BuildContext
    {
        const Allocator<std::byte> allocator;
        const compute::CudaStream cudaStream;
        const Exec exec;

        sah_kd_tree::Builder<BuilderContext> builder{allocator, exec};
        sah_kd_tree::Projection<BuilderContext> x{allocator, exec}, y{allocator, exec}, z{allocator, exec};

        sah_kd_tree::Triangle<BuilderContext> triangle{allocator, exec};

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

template TreePtr build<ThrustDeviceSystem::CUDA>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
