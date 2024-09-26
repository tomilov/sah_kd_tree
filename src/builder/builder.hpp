#pragma once

#include <builder/fwd.hpp>
#include <compute/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fd.hpp>
#include <utils/noncopyable.hpp>

#include <functional>
#include <memory>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <builder/builder_export.h>

namespace builder
{

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

    Tree(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress);
    Tree(Tree &&) noexcept;
    ~Tree();

    [[nodiscard]] const Settings & getSettings() const &;
    [[nodiscard]] const compute::CudaDevice & getCudaDevice() const &;
    [[nodiscard]] scene_data::SceneDataPtr getSceneData() const;

    [[nodiscard]] size_t getTriangleCount() const;
    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] size_t getPolygonCount() const;
    [[nodiscard]] size_t getNodeCount() const;

    [[nodiscard]] size_t getDataSize() const;
    [[nodiscard]] size_t getDataAlignment() const;
    [[nodiscard]] size_t getAllocationSize() const;

    [[nodiscard]] size_t getTriangleOffset() const;
    [[nodiscard]] size_t getPolygonOffset() const;
    [[nodiscard]] size_t getNodeOffset() const;
    [[nodiscard]] size_t getNodeParentOffset() const;

    [[nodiscard]] bool isEmpty() const;
    [[nodiscard]] utils::Fd getFd() &&;
    [[nodiscard]] utils::Fd cloneFd() const &;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace builder
