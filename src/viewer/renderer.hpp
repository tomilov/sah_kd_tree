#pragma once

#include <engine/fwd.hpp>
#include <format/glm.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/scenes.hpp>

#include <fmt/base.h>
#include <glm/ext/quaternion_float.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace viewer
{
class Engine;
struct Scene;

struct FrameSettings
{
    // bool clipByDiscard = false;
    bool useOffscreenTexture = false;
    glm::mat4 transform2D{1.0f};
    float alpha = 1.0f;
    float zNear = 1E-3f;
    float zFar = 1E3f;
    glm::vec3 position{0.0f};
    glm::quat orientation = glm::quat_identity<glm::quat::value_type, glm::defaultp>();
    vk::Rect2D scissor = {};
    vk::Viewport viewport = {};
    float width = 0.0f;   // TODO: int
    float height = 0.0f;  // TODO: int
    float fov = glm::half_pi<float>();

    bool operator==(const FrameSettings &) const = default;
    bool operator!=(const FrameSettings &) const = default;
};

class Renderer : utils::NonCopyable
{
public:
    Renderer(const engine::Context & context, const Engine & engine, uint32_t framesInFlight);
    ~Renderer();

    [[nodiscard]] uint32_t getFramesInFlight() const;

    void setFrameSettings(const FrameSettings & frameSettings);

    void setScene(std::shared_ptr<const Scene> scene);
    void unsetScene();
    [[nodiscard]] const std::shared_ptr<const Scene> & getScene() const &;

    void advance(uint32_t currentFrameSlot);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot);

private:
    struct Impl;

    static constexpr size_t kSize = 520;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer

template<>
struct fmt::formatter<viewer::FrameSettings> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const viewer::FrameSettings & frameSettings, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), "{{.useOffscreenTexture = {}, .transform2D = {}, .alpha = {}, .zNear = {}, .zFar = {}, .position = {}, .orientation = {}, .scissor = {}, .viewport = {}, .width = {}, .height = {}, .fov = {}}}",
                              frameSettings.useOffscreenTexture, frameSettings.transform2D, frameSettings.alpha, frameSettings.zNear, frameSettings.zFar, frameSettings.position, frameSettings.orientation, frameSettings.scissor,
                              frameSettings.viewport, frameSettings.width, frameSettings.height, frameSettings.fov);
    }
};
