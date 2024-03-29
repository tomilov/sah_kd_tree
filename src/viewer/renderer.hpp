#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <glm/ext/quaternion_float.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <string_view>

#include <cstdint>

namespace viewer
{
class Scene;

struct FrameSettings
{
    bool useOffscreenTexture = false;
    glm::mat2 transform2D{1.0f};
    float alpha = 1.0f;
    float zNear = 1E-3f;
    float zFar = 1E3f;
    glm::vec3 position{0.0f};
    glm::quat orientation = glm::quat_identity<glm::quat::value_type, glm::defaultp>();
    float t = 0.0f;
    vk::Rect2D scissor = {};
    vk::Viewport viewport = {};
    float width = 0.0f;
    float height = 0.0f;
    float fov = glm::half_pi<float>();

    bool operator==(const FrameSettings &) const = default;
    bool operator!=(const FrameSettings &) const = default;
};

class Renderer : utils::NonCopyable
{
public:
    Renderer(const engine::Context & context, uint32_t framesInFlight);
    ~Renderer();

    void setScene(std::shared_ptr<const Scene> scene);
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    [[nodiscard]] bool updateRenderPass(vk::RenderPass renderPass);
    void render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const;

private:
    struct Impl;

    static constexpr size_t kSize = 184;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
