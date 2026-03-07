#pragma once

#include <builder/fwd.hpp>
#include <compute/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>

#include <functional>
#include <optional>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{

enum class ThrustDeviceSystem
{
    eDefault,
    eCUDA,
    eTBB,
    eOMP,
    eCPP,
};

struct Settings
{
    const float emptinessFactor;
    const float traversalCost;
    const float intersectionCost;
    const uint32_t maxTreeDepth;

    auto operator<=>(const Settings &) const = default;
};

struct Tree
{
    const Settings settings;
    const compute::CudaDevice & cudaDevice;
    const scene_data::SceneDataWeakPtr sceneData;

    size_t dataSize = 0;
    size_t dataAlignment = 0;
    size_t allocationSize = 0;

    size_t triangleCount = 0;
    std::vector<size_t> layerSizes = {};
    size_t polygonCount = 0;
    size_t nodeCount = 0;

    size_t triangleOffset = 0;
    size_t polygonOffset = 0;
    size_t nodeOffset = 0;
    size_t nodeParentOffset = 0;

    utils::Fd fd = {};

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<Tree>::checkTraits();
    }
};

std::optional<Tree> build(ThrustDeviceSystem deviceSystem, const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;

inline std::optional<Tree> build(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
{
    return build(ThrustDeviceSystem::eDefault, settings, cudaDevice, sceneData, progress);
}

}  // namespace builder
