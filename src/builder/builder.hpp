#pragma once

#include <builder/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <array>
#include <functional>
#include <memory>
#include <optional>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{
class CudaDevice;

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

    bool build(const std::function<bool()> & cancel);

private:
    friend Builder;
    struct Impl;

    std::unique_ptr<Impl> impl_;

    Tree(const Settings & settings, const CudaDevice & cudaDevice, const scene_data::SceneData & sceneData);

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class BUILDER_EXPORT Builder : utils::OneTime<Builder>
{
public:
    struct Settings
    {
        using DeviceUuidType = std::array<std::byte, 16>;

        bool skipDeviceCheck = false;  // first device will be selected
        DeviceUuidType deviceUuid;
        size_t minAlignment;
    };

    Builder(const Settings & settings);
    Builder(Builder &&) noexcept;
    ~Builder();

    std::optional<Tree> build(const Tree::Settings & treeSettings, const scene_data::SceneData & sceneData, const std::function<bool()> & cancel) const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace builder
