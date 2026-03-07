#pragma once

#include <builder/fwd.hpp>
#include <soft_renderer/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <gli/texture2d.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/fwd.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <string_view>

#include <soft_renderer/soft_renderer_export.h>

namespace soft_renderer
{

struct FrameSettings
{
    glm::vec3 position{0.0f};
    glm::quat orientation = glm::quat_identity<glm::quat::value_type, glm::defaultp>();
    glm::float32 fov = glm::half_pi<float>();
    glm::float32 zNear = 1E-3f;
    glm::float32 zFar = 1E3f;

    bool operator==(const FrameSettings &) const = default;
    bool operator!=(const FrameSettings &) const = default;
};

class SOFT_RENDERER_EXPORT SoftRenderer : utils::OneTime<SoftRenderer>
{
public:
    static inline gli::format kTargetFormat = gli::format::FORMAT_RGBA8_UNORM_PACK8;

    using PixelType = glm::u8vec4;

    SoftRenderer(std::string_view name, const glm::vec4 & clearColor);
    SoftRenderer(SoftRenderer &&) noexcept;
    ~SoftRenderer();

    void setTree(builder::Tree && builderTree);
    bool hasTree() const;

    void render(const FrameSettings & frameSettings, gli::texture2d & target) const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace soft_renderer
