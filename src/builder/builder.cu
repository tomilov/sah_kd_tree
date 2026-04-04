#include <builder/builder.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <initializer_list>
#include <memory>
#include <utility>

template struct utils::OneTime<builder::Tree>::CheckTraits;

namespace builder
{

decltype(&build<>) getBuild(size_t i)
{
    std::initializer_list<decltype(&build<>)> builds = {
        &build<ThrustDeviceSystem::Default>,  //
        &build<ThrustDeviceSystem::CPP>,      //
        &build<ThrustDeviceSystem::OMP>,      //
        &build<ThrustDeviceSystem::TBB>,      //
        &build<ThrustDeviceSystem::CUDA>,     //
    };
    return (i < std::size(builds)) ? builds.begin()[i] : nullptr;
}

void TreeDeleter::operator()(Tree * tree) const noexcept
{
    std::default_delete<Tree>{}(tree);
}

TreePtr makeTreePtr(Tree && tree)
{
    return {new Tree{std::move(tree)}, TreeDeleter{}};
}

template<>
struct BuilderContext<ThrustDeviceSystem::Default>
{
    using Traits = sah_kd_tree::DefaultTraits;

    template<typename T>
    using Vector = Traits::template Vector<T>;

    struct TreeContext
    {
        typename Traits::Exec exec;

        struct Index
        {
            Vector<Traits::U> a, b, c;
        } index;

        struct Vertex
        {
            Vector<Traits::F> x, y, z;
        } vertex;

        sah_kd_tree::Tree<sah_kd_tree::DefaultTraits> tree;

        void synchronize() const
        {}
    };

    struct BuildContext
    {
        sah_kd_tree::Projection<sah_kd_tree::DefaultTraits> x, y, z;
        sah_kd_tree::Builder<sah_kd_tree::DefaultTraits> builder;

        explicit BuildContext(const TreeContext &)
        {}
    };
};

template TreePtr build<ThrustDeviceSystem::Default>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
