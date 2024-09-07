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
    Tree(Tree &&) noexcept;
    ~Tree();

    [[nodiscard]] builder::TreePtr getBuilderTree() const;

    [[nodiscard]] uint32_t getTriangleCount() const;
    [[nodiscard]] const std::vector<size_t> & getLayerSizes() const &;
    [[nodiscard]] uint32_t getPolygonCount() const;
    [[nodiscard]] uint32_t getNodeCount() const;

    [[nodiscard]] vk::DeviceSize getDataSize() const;
    [[nodiscard]] vk::DeviceSize getDataAlignment() const;
    [[nodiscard]] vk::DeviceSize getAllocationSize() const;
    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;

    [[nodiscard]] vk::DeviceAddress getTriangleAddress() const &;
    [[nodiscard]] vk::DeviceAddress getPolygonAddress() const &;
    [[nodiscard]] vk::DeviceAddress getNodeAddress() const &;
    [[nodiscard]] vk::DeviceAddress getNodeParentAddress() const &;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace viewer
