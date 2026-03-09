#pragma once

#include <builder/fwd.hpp>
#include <engine/fwd.hpp>
#include <format/glm.hpp>
#include <format/vulkan.hpp>
#include <scene_data/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/scenes.hpp>

#include <fmt/base.h>
#include <glm/ext/quaternion_float.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.hpp>

#include <memory>
#include <string_view>

#include <cstdint>

namespace viewer
{
class Engine;
struct Scene;

struct FrameSettings
{
    bool traceSahKdTree = false;
    bool useOffscreenTexture = false;
    bool discardInvisible = false;
    bool wireframe = false;

    glm::vec3 position{0.0f};
    glm::quat orientation = glm::quat_identity<glm::quat::value_type, glm::defaultp>();
    float fov = glm::half_pi<float>();
    float zNear = 1E-3f;
    float zFar = 1E3f;

    float alpha = 1.0f;
    float width = 0.0f;
    float height = 0.0f;
    vk::Viewport viewport = {};
    vk::Rect2D scissor = {};
    glm::mat4 windowMvp{1.0f};

    glm::vec4 clearColor{0.0f, 0.0f, 0.0f, 1.0f};

    bool operator==(const FrameSettings &) const = default;
    bool operator!=(const FrameSettings &) const = default;

    [[nodiscard]] vk::Extent2D getFramebufferSize() const;
};

class Renderer : utils::NonCopyable
{
public:
    Renderer(
        std::string_view name,
        const engine::Context & context,
        const Engine & engine,
        uint32_t framesInFlight);
    ~Renderer();

    [[nodiscard]] uint32_t getFramesInFlight() const;

    void setFrameSettings(const FrameSettings & frameSettings);

    void setScene(scene_data::SceneDataPtr sceneData);
    void unsetScene();
    [[nodiscard]] const scene_data::SceneDataPtr & getScene() const &;

    void setTree(builder::TreePtr builderTree);

    void advance(
        vk::CommandBuffer commandBuffer,
        uint32_t currentFrameSlot);
    void render(
        vk::CommandBuffer commandBuffer,
        vk::RenderPass renderPass,
        bool isRenderPassFormatChanged,
        uint32_t currentFrameSlot);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer

template<>
struct fmt::formatter<viewer::FrameSettings> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        const viewer::FrameSettings & frameSettings,
        FormatContext & ctx) const
    {
        constexpr auto fmtString
            = "{{"  //
              ".useOffscreenTexture = {}, "
              ".discardInvisible = {}, "
              ".wireframe = {}, "
              ".position = {}, "
              ".orientation = {}, "
              ".fov = {}, "
              ".zNear = {}, "
              ".zFar = {}, "
              ".alpha = {}, "
              ".width = {}, "
              ".height = {}, "
              ".viewport = {}, "
              ".scissor = {}, "
              ".windowMvp = {}, "
              ".clearColor = {}"
              "}}";
        return fmt::format_to(
            ctx.out(),                          //
            fmtString,                          //
            frameSettings.useOffscreenTexture,  //
            frameSettings.discardInvisible,     //
            frameSettings.wireframe,            //
            frameSettings.position,             //
            frameSettings.orientation,          //
            frameSettings.fov,                  //
            frameSettings.zNear,                //
            frameSettings.zFar,                 //
            frameSettings.alpha,                //
            frameSettings.width,                //
            frameSettings.height,               //
            frameSettings.viewport,             //
            frameSettings.scissor,              //
            frameSettings.windowMvp,            //
            frameSettings.clearColor);          //
    }
};
