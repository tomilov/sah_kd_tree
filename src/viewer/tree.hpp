#pragma once

#include <builder/fwd.hpp>
#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <span>

namespace viewer
{

class Tree : utils::OneTime<Tree>
{
public:
    explicit Tree(const engine::Context & context, const builder::TreePtr & builderTree, std::span<const uint32_t> queueFamilies);
    ~Tree();

    [[nodiscard]] builder::TreePtr getBuilderTree() const;

    [[nodiscard]] vk::DeviceSize getAllocationSize() const;
    [[nodiscard]] vk::DeviceSize getDataSize() const;

    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] uint32_t getPolygonCount() const;
    [[nodiscard]] uint32_t getNodeCount() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer
