#pragma once

#include <builder/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>

#include <glm/fwd.hpp>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{
class CudaDevice;
using DeviceUuidType = std::array<std::byte, 16>;

class BUILDER_EXPORT Tree : utils::OneTime<Tree>
{
public:
    struct Settings
    {
        float emptinessFactor;
        float traversalCost;
        float intersectionCost;
        uint32_t maxDepth;

        auto operator<=>(const Settings &) const = default;
    };

    Tree(Tree &&) noexcept;
    ~Tree();

    [[nodiscard]] const Settings & getSettings() const &;
    [[nodiscard]] scene_data::SceneDataPtr getSceneData() const;

    [[nodiscard]] bool isEmpty() const;
    [[nodiscard]] utils::Fd getFd() &&;
    [[nodiscard]] utils::Fd cloneFd() const &;
    [[nodiscard]] size_t getAllocationSize() const;
    [[nodiscard]] size_t getDataSize() const;

    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] size_t getPolygonCount() const;
    [[nodiscard]] size_t getNodeCount() const;

private:
    friend Builder;
    struct Impl;

    std::unique_ptr<Impl> impl_;

    Tree(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(float progressValue, const std::string & progressText)> & progress);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class BUILDER_EXPORT Builder : utils::OneTime<Builder>
{
public:
    Builder(const std::optional<DeviceUuidType> & deviceUuid);
    Builder(Builder &&) noexcept;
    ~Builder();

    std::optional<Tree> build(const Tree::Settings & treeSettings, const scene_data::SceneDataPtr & sceneData, const std::function<bool(float progressValue, const std::string & progressText)> & progress) const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace builder
