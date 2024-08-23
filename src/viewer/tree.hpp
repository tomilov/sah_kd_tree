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
    // eStorageBuffer | eShaderDeviceAddress? eAccelerationStructureBuildInputReadOnlyKHR?
    explicit Tree(const engine::Context & context, const builder::TreePtr & builderTree, vk::BufferUsageFlags usage, std::span<const uint32_t> queueFamilies);
    ~Tree();

    [[nodiscard]] builder::TreePtr getBuilderTree() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer
