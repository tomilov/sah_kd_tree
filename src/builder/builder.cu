#include <builder/builder.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <memory>
#include <utility>

namespace builder
{

void TreeDeleter::operator()(Tree * tree) const noexcept
{
    return std::default_delete<Tree>{}(tree);
}

TreePtr makeTreePtr(Tree && tree)
{
    return {new Tree{std::move(tree)}, TreeDeleter{}};
}

struct ThrustDeviceSystemDefault
{
    using Traits = sah_kd_tree::DefaultTraits;

    struct TreeContext
    {
        sah_kd_tree::Tree<sah_kd_tree::DefaultTraits> tree;
    };

    struct BuildContext
    {
        sah_kd_tree::Builder<sah_kd_tree::DefaultTraits> builder;
        sah_kd_tree::Projection<sah_kd_tree::DefaultTraits> x, y, z;

        sah_kd_tree::Triangle<sah_kd_tree::DefaultTraits> triangle;

        explicit BuildContext(const TreeContext &)
        {}
    };
};

template class TreeBuildContext<ThrustDeviceSystemDefault>;
template TreePtr build<ThrustDeviceSystemDefault>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress);

}  // namespace builder
