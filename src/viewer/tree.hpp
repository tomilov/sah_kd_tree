#pragma once

#include <builder/fwd.hpp>
#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string_view>

#include <cstdint>

namespace viewer
{

class Tree : utils::OneTime<Tree>
{
public:
    explicit Tree(std::string_view name, const engine::Context & context, const builder::TreePtr & builderTree);
    ~Tree();

    [[nodiscard]] builder::TreePtr getBuilderTree() const;

    [[nodiscard]] vk::DeviceSize getDataSize() const;
    [[nodiscard]] vk::DeviceSize getAllocationSize() const;

    [[nodiscard]] uint32_t getTriangleCount() const;
    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] uint32_t getPolygonCount() const;
    [[nodiscard]] uint32_t getNodeCount() const;

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer
