#pragma once

#include <builder/fwd.hpp>
#include <softrenderer/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <gli/texture2d.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <string_view>

#include <softrenderer/softrenderer_export.h>

namespace softrenderer
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

class SOFTRENDERER_EXPORT SoftRenderer : utils::OneTime<SoftRenderer>
{
public:
    explicit SoftRenderer(std::string_view name, glm::vec4 clearColor);

    void unsetTree();
    void updateTree(const builder::TreePtr & tree);
    [[nodiscard]] const builder::TreePtr & getTree() const &;

    void render(const FrameSettings & frameSettings, gli::texture2d & target) const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace softrenderer
