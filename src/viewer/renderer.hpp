#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <glm/ext/quaternion_float.hpp>
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
class SceneManager;

struct FrameSettings
{
    glm::vec3 position{0.0f};
    glm::quat orientation = glm::quat_identity<glm::quat::value_type, glm::defaultp>();
    float t = 0.0f;
    float alpha = 1.0f;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    glm::mat3 transform2D{1.0f};
    const float fov = 90.0f;
    const float zNear = 1E-3f;
    const float zFar = 1E3f;
    const float scale = 1.0f;
};

class Renderer : utils::NonCopyable
{
public:
    Renderer(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager);
    ~Renderer();

    [[nodiscard]] const std::filesystem::path & getScenePath() const;

    void advance(uint32_t framesInFlight);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, uint32_t framesInFlight, const FrameSettings & frameSettings);

private:
    struct Impl;

    static constexpr size_t kSize = 144;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
