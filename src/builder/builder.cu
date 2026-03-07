#include "builder/builder.cuh"

#include <utils/assert.hpp>

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

std::optional<Tree> build(ThrustDeviceSystem deviceSystem, const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
{
    std::unique_ptr<TreeBuildContextBase> treeBuildContext;
    switch (deviceSystem) {
    case ThrustDeviceSystem::eCUDA: {
        treeBuildContext = std::make_unique<TreeBuildContextCUDA>(settings, cudaDevice, sceneData);
        break;
    }
    case ThrustDeviceSystem::eTBB: {
        treeBuildContext = std::make_unique<TreeBuildContextTBB>(settings, cudaDevice, sceneData);
        break;
    }
    case ThrustDeviceSystem::eOMP: {
        treeBuildContext = std::make_unique<TreeBuildContextOMP>(settings, cudaDevice, sceneData);
        break;
    }
    case ThrustDeviceSystem::eCPP: {
        treeBuildContext = std::make_unique<TreeBuildContextCPP>(settings, cudaDevice, sceneData);
        break;
    }
    }
    INVARIANT(treeBuildContext, "");
    if (!treeBuildContext->build(progress)) {
        return {};
    }
    return std::move(*treeBuildContext);
}

}  // namespace builder
