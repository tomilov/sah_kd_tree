#include <builder/builder.cuh>
#include <sah_kd_tree/sah_kd_tree_inline.cuh>

#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/cpp/memory_resource.h>
#include <thrust/system/cpp/vector.h>

namespace builder
{

struct ThrustDeviceSystemCPP
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    template<typename T>
    using Allocator = thrust::cpp::allocator<T>;
    template<typename T>
    using Vector = thrust::cpp::vector<T, Allocator<T>>;

    struct TreeContext
    {
        sah_kd_tree::Tree<ThrustDeviceSystemCPP> tree;
    };

    using Exec = decltype(thrust::cpp::par);
    using Progress = sah_kd_tree::DefaultTraits::Progress;

    struct BuildContext
    {
        sah_kd_tree::Builder<ThrustDeviceSystemCPP> builder;
        sah_kd_tree::Projection<ThrustDeviceSystemCPP> x, y, z;

        sah_kd_tree::Triangle<ThrustDeviceSystemCPP> triangle;

        explicit BuildContext(const TreeContext &)
        {}
    };
};

template class TreeBuildContext<ThrustDeviceSystemCPP>;
template TreePtr build<ThrustDeviceSystemCPP>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
