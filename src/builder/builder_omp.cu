#include <builder/builder.cuh>
#include <sah_kd_tree/sah_kd_tree_inline.cuh>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/memory.h>
#include <thrust/system/omp/memory_resource.h>
#include <thrust/system/omp/vector.h>

namespace builder
{

struct ThrustDeviceSystemOMP
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    template<typename T>
    using Allocator = thrust::omp::allocator<T>;
    template<typename T>
    using Vector = thrust::omp::vector<T, Allocator<T>>;

    struct TreeContext
    {
        sah_kd_tree::Tree<ThrustDeviceSystemOMP> tree;
    };

    using Exec = decltype(thrust::omp::par);
    using Progress = sah_kd_tree::DefaultTraits::Progress;

    struct BuildContext
    {
        sah_kd_tree::Builder<ThrustDeviceSystemOMP> builder;
        sah_kd_tree::Projection<ThrustDeviceSystemOMP> x, y, z;

        sah_kd_tree::Triangle<ThrustDeviceSystemOMP> triangle;

        explicit BuildContext(const TreeContext &)
        {}
    };
};

template class TreeBuildContext<ThrustDeviceSystemOMP>;
template TreePtr build<ThrustDeviceSystemOMP>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
