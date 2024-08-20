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
#include <vector>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{
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

    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] size_t getPolygonCount() const;
    [[nodiscard]] size_t getNodeCount() const;
    [[nodiscard]] utils::Fd getFd() &&;

private:
    friend Builder;
    struct Impl;

    std::unique_ptr<Impl> impl_;

    Tree(const Settings & settings, const std::optional<DeviceUuidType> & deviceUuidType, const scene_data::SceneData & sceneData);

    bool build(const std::function<bool()> & cancel);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class BUILDER_EXPORT Builder : utils::OneTime<Builder>
{
public:
    struct Settings
    {
        std::optional<DeviceUuidType> deviceUuid;
    };

    Builder(const Settings & settings);
    Builder(Builder &&) noexcept;
    ~Builder();

    std::optional<Tree> build(const Tree::Settings & treeSettings, const scene_data::SceneData & sceneData, const std::function<bool()> & cancel) const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace builder
