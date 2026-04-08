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
    explicit Tree(
        utils::Name name,
        const engine::Context & context,
        builder::Tree && builderTree);
    Tree(Tree &&) noexcept;
    ~Tree();

    [[nodiscard]] uint32_t getTriangleCount() const;
    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] uint32_t getPolygonCount() const;
    [[nodiscard]] uint32_t getNodeCount() const;

    [[nodiscard]] vk::DeviceSize getDataSize() const;
    [[nodiscard]] vk::DeviceSize getDataAlignment() const;
    [[nodiscard]] vk::DeviceSize getAllocationSize() const;
    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;

    [[nodiscard]] vk::DeviceAddress getIndexAddress() const &;
    [[nodiscard]] vk::DeviceAddress getVertexAddress() const &;
    [[nodiscard]] vk::DeviceAddress getPolygonAddress() const &;
    [[nodiscard]] vk::DeviceAddress getNodeAddress() const &;
    [[nodiscard]] vk::DeviceAddress getNodeParentAddress() const &;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer
