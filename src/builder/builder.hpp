#pragma once

#include <builder/fwd.hpp>
#include <compute/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>

#include <concepts>
#include <functional>
#include <initializer_list>
#include <optional>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{

struct ThrustDeviceSystemDefault;
struct ThrustDeviceSystemCPP;
struct ThrustDeviceSystemOMP;
struct ThrustDeviceSystemTBB;
struct ThrustDeviceSystemCUDA;

template<typename T>
concept ThrustDeviceSystem = std::same_as<T, ThrustDeviceSystemDefault> || std::same_as<T, ThrustDeviceSystemCPP> || std::same_as<T, ThrustDeviceSystemOMP> || std::same_as<T, ThrustDeviceSystemTBB> || std::same_as<T, ThrustDeviceSystemCUDA>;

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
    const scene_data::SceneDataPtr sceneData;

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

    std::optional<utils::Fd> fd = std::nullopt;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<Tree>::checkTraits();
    }
};

template<ThrustDeviceSystem Traits>
[[nodiscard]] TreePtr build(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress);

extern template TreePtr build<ThrustDeviceSystemDefault>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;
extern template TreePtr build<ThrustDeviceSystemCPP>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;
extern template TreePtr build<ThrustDeviceSystemOMP>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;
extern template TreePtr build<ThrustDeviceSystemTBB>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;
extern template TreePtr build<ThrustDeviceSystemCUDA>(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress) BUILDER_EXPORT;

constexpr auto getBuild(size_t i)
{
    // clang-format off
    auto builds = {
        &builder::build<builder::ThrustDeviceSystemDefault>,
        &builder::build<builder::ThrustDeviceSystemCPP>,
        &builder::build<builder::ThrustDeviceSystemOMP>,
        &builder::build<builder::ThrustDeviceSystemTBB>,
        &builder::build<builder::ThrustDeviceSystemCUDA>,
    };
    // clang-format on
    return (i < std::size(builds)) ? builds.begin()[i] : nullptr;
}

}  // namespace builder
