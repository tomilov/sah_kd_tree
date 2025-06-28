#include <builder/builder.hpp>
#include <scene_data/scene_data.hpp>
#include <soft_renderer/soft_renderer.hpp>
#include <soft_renderer/tree.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <gli/type.hpp>
#include <glm/common.hpp>
#include <glm/ext/vector_bool3.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda.h>

namespace soft_renderer
{
namespace
{

bool rayTriangleIntersect(const Ray & ray, Hit & hit, const scene_data::Triangle & triangle, glm::float32 tNear, glm::float32 tFar)
{
    // TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
    const glm::vec3 v1v0 = triangle.b - triangle.a;
    const glm::vec3 v2v0 = triangle.c - triangle.a;
    const glm::vec3 rov0 = ray.pos - triangle.a;
    const glm::vec3 n = glm::cross(v1v0, v2v0);
    const float d = 1.0f / glm::dot(ray.dir, n);
    hit.t = d * glm::dot(-n, rov0);
    if ((hit.t < tNear) || (tFar < hit.t)) {
        return false;
    }
    const glm::vec3 q = glm::cross(rov0, ray.dir);
    hit.uv.x = d * glm::dot(-q, v2v0);
    hit.uv.y = d * glm::dot(q, v1v0);
    return !((hit.uv.x < 0.0f) || (hit.uv.y < 0.0f) || (hit.uv.x + hit.uv.y > 1.0f));
}

bool intersectTriangle(const Ray & ray, Hit & hit, const scene_data::Triangle & triangle, glm::float32 tNear, glm::float32 tFar /*, glm::vec3 & outNormal*/)
{
    // Can be shared between all triangles
    glm::vec3 centerU = ray.dir;
    glm::vec3 centerV = cross(ray.dir, ray.pos);
    // Constant
    glm::vec3 v0U = (triangle.b - triangle.a);
    glm::vec3 v0V = cross(triangle.b, triangle.a);
    glm::vec3 v1U = (triangle.c - triangle.b);
    glm::vec3 v1V = cross(triangle.c, triangle.b);
    glm::vec3 v2U = (triangle.a - triangle.c);
    glm::vec3 v2V = cross(triangle.a, triangle.c);
    // 6 dot intersection test
    if ((glm::dot(v0U, centerV) + glm::dot(v0V, centerU) > 0.0f) && (glm::dot(v1U, centerV) + glm::dot(v1V, centerU) > 0.0f) && (glm::dot(v2U, centerV) + glm::dot(v2V, centerU) > 0.0f)) {
        glm::vec3 normal = glm::normalize(glm::cross(triangle.b - triangle.a, triangle.c - triangle.a));
        glm::float32 t = glm::dot(ray.pos - triangle.a, normal) / glm::dot(ray.dir, normal);
        if (t < hit.t) {
            // outNormal = normal;
            hit.t = t;
            return true;
        }
    }
    return false;
}

bool intersectSphere(const Ray & ray, const glm::vec3 & center, glm::float32 radius)
{
    glm::vec3 oc = center - ray.pos;
    glm::float32 l = glm::dot(ray.dir, oc);
    if (l < 0.0f) {
        return false;
    }
    glm::vec3 ll = ray.dir * l;
    return radius * radius > glm::dot(oc, oc) - glm::dot(ll, ll);
}

}  // namespace

struct SoftRenderer::Impl
{
    const std::string name;
    const glm::vec4 clearColor;

    std::vector<scene_data::Triangle> triangles;
    std::vector<glm::uint> polygons;
    std::vector<Node> nodes;
    std::vector<glm::uint> nodeParents;

    Impl(std::string_view name, const glm::vec4 & clearColor)
        : name{name}
        , clearColor{clearColor}
    {}

    [[nodiscard]] glm::uint findNode(glm::uint nodeIndex, const Ray & ray) const
    {
        for (;;) {
            const Node & node = nodes.at(nodeIndex);
            const glm::int32 splitDimension = node.splitDimension;
            if (splitDimension < 0) {
                return nodeIndex;
            }
            if (ray.pos[splitDimension] < node.splitPos) {
                nodeIndex = node.leftChild;
            } else {
                nodeIndex = node.rightChild;
            }
        }
    }

    [[nodiscard]] bool traceRay(glm::uint nodeIndex, const Ray & ray, Hit & hit) const
    {
        if ((false)) {
            const scene_data::Triangle triangle = {
                .a = glm::vec3{0.0f, 0.0f, 0.0f},
                .b = glm::vec3{0.0f, 1.0f, 0.0f},
                .c = glm::vec3{1.0f, 0.0f, 0.0f},
            };
            constexpr glm::float32 kNear = 0.0f;
            constexpr glm::float32 kFar = 100.0f;
            return rayTriangleIntersect(ray, hit, triangle, kNear, kFar);
        }
        const glm::vec3 invDir = 1.0f / ray.dir;
        const glm::bvec3 corner = glm::lessThan(invDir, glm::vec3{0.0f});
        glm::vec3 aabbHitT = (glm::mix(nodes.at(nodeIndex).aabbMin, nodes.at(nodeIndex).aabbMax, corner) - ray.pos) * invDir;
        glm::float32 tMin = glm::max(aabbHitT.x, glm::max(aabbHitT.y, aabbHitT.z));
        do {
            const glm::float32 tNear = glm::max(0.0f, tMin);
            for (;;) {
                const Node & node = nodes.at(nodeIndex);
                const glm::int32 splitDimension = node.splitDimension;
                if (splitDimension < 0) {
                    break;
                }
                if (corner[splitDimension] == ((node.splitPos - ray.pos[splitDimension]) * invDir[splitDimension] < tNear)) {
                    nodeIndex = node.leftChild;
                } else {
                    nodeIndex = node.rightChild;
                }
            }
            const Node & node = nodes.at(nodeIndex);
            aabbHitT = (glm::mix(node.aabbMax, node.aabbMin, corner) - ray.pos) * invDir;
            const glm::float32 tMax = glm::min(aabbHitT.x, glm::min(aabbHitT.y, aabbHitT.z));
            if (tMin > tMax) {
                return false;
            }
            const glm::uint polygonStart = node.leftChild;
            const glm::uint polygonEnd = polygonStart + node.rightChild;
            for (glm::uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
                const glm::float32 tFar = glm::min(hit.t, tMax);
                Hit closerHit;
                closerHit.triangle = polygons.at(polygon);
                if (rayTriangleIntersect(ray, closerHit, triangles.at(closerHit.triangle), tNear, tFar)) {
                    if (closerHit.t < hit.t) {
                        hit = closerHit;
                    }
                }
            }
            if (hit.t <= tMax) {
                return true;
            }
            tMin = tMax;
            const glm::ivec3 indices = glm::mix(glm::ivec3(0), glm::ivec3(0, 1, 2), glm::equal(aabbHitT, glm::vec3(tMax)));
            const glm::int32 ropeDirection = glm::max(indices.x, glm::max(indices.y, indices.z));
            nodeIndex = corner[ropeDirection] ? node.leftRope[ropeDirection] : node.rightRope[ropeDirection];
        } while (nodeIndex != 0u);
        return false;
    }
};

SoftRenderer::SoftRenderer(std::string_view name, const glm::vec4 & clearColor)
    : impl_{std::make_unique<Impl>(name, clearColor)}
{}

SoftRenderer::~SoftRenderer() = default;

SoftRenderer::SoftRenderer(SoftRenderer &&) noexcept = default;

void SoftRenderer::setTree(builder::Tree && builderTree)
{
    importTree(std::move(builderTree), impl_->triangles, impl_->polygons, impl_->nodes, impl_->nodeParents);
}

void SoftRenderer::render(const FrameSettings & frameSettings, gli::texture2d & target) const
{
    INVARIANT(!target.empty(), "");
    INVARIANT(target.format() == kTargetFormat, "{}", fmt::underlying(target.format()));
    constexpr size_t kLevel = 0;
    // target.clear(PixelType(impl_->clearColor));
    const gli::extent2d extent = target.extent();
    const glm::float32 dy = glm::tan(frameSettings.fov * 0.5f);
    const glm::float32 dx = dy * (utils::safeCast<glm::float32>(extent.x) / utils::safeCast<glm::float32>(extent.y));
    const glm::vec3 leftTop = glm::rotate(frameSettings.orientation, glm::vec3{-dx, dy, 1.0f});
    const glm::vec3 rightTop = glm::rotate(frameSettings.orientation, glm::vec3{dx, dy, 1.0f});
    const glm::vec3 leftBottom = glm::rotate(frameSettings.orientation, glm::vec3{-dx, -dy, 1.0f});
    const glm::vec3 rightBottom = glm::rotate(frameSettings.orientation, glm::vec3{dx, -dy, 1.0f});
    const glm::vec2 invExtent = 1.0f / glm::vec2{extent};
    Ray ray;
    ray.pos = frameSettings.position;
    Hit hit;
    hit.t = std::numeric_limits<glm::float32>::infinity();
    const glm::uint nodeIndex = impl_->findNode(kRootNodeIndex, ray);
    for (gli::int32 y = 0; y < extent.y; ++y) {
        if (y + y != extent.y) {
            //continue;
        }
        const glm::float32 locY = (utils::safeCast<glm::float32>(y) + 0.5f) * invExtent.y;
        const glm::vec3 left = glm::mix(leftBottom, leftTop, locY);
        const glm::vec3 right = glm::mix(rightBottom, rightTop, locY);
        for (gli::int32 x = 0; x < extent.x; ++x) {
            if (x + x != extent.x) {
                //continue;
            }
            const glm::float32 locX = (utils::safeCast<glm::float32>(x) + 0.5f) * invExtent.x;
            ray.dir = glm::normalize(glm::mix(left, right, locX));
            glm::vec4 texel;
            if ((true)) {
                if (impl_->traceRay(nodeIndex, ray, hit)) {
                    texel = glm::vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv, 1.0f);
                } else {
                    texel = impl_->clearColor;
                }
            } else {
                if (intersectSphere(ray, glm::vec3{}, 0.5f)) {
                    texel = glm::vec4{1.0f, 0.0f, 0.0f, 1.0f};
                } else {
                    texel = impl_->clearColor;
                }
            }
            constexpr auto kScale = static_cast<glm::vec4::value_type>(std::numeric_limits<PixelType::value_type>::max());
            target.store(gli::extent2d{x, y}, kLevel, PixelType(glm::clamp(texel, glm::vec4{0.0f}, glm::vec4{1.0f}) * kScale));
        }
    }
}

}  // namespace soft_renderer
