#include <builder/builder.cuh>
#include <sah_kd_tree/sah_kd_tree_inline.cuh>

#include <thrust/system/tbb/execution_policy.h>
#include <thrust/system/tbb/memory.h>
#include <thrust/system/tbb/memory_resource.h>
#include <thrust/system/tbb/vector.h>

namespace builder
{

template<>
struct BuilderContext<ThrustDeviceSystem::TBB>
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    template<typename T>
    using Allocator = thrust::tbb::allocator<T>;
    template<typename T>
    using Vector = thrust::tbb::vector<T, Allocator<T>>;
    using ComponentIterator = typename Vector<F>::const_pointer;

    struct TreeContext
    {
        sah_kd_tree::Tree<BuilderContext> tree;
    };

    using Exec = decltype(thrust::tbb::par);
    using Progress = sah_kd_tree::DefaultTraits::Progress;

    struct BuildContext
    {
        sah_kd_tree::Builder<BuilderContext> builder;
        sah_kd_tree::Projection<BuilderContext> x, y, z;

        sah_kd_tree::Triangle<BuilderContext> triangle;

        explicit BuildContext(const TreeContext &)
        {}
    };
};

template TreePtr build<ThrustDeviceSystem::TBB>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
