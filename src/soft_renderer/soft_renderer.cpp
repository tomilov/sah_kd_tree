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

#include <cmath>

#include <cuda.h>

namespace soft_renderer
{
namespace
{

const glm::float32 kUlp = std::nextafter(0.0f, 1.0f);
constexpr glm::float32 kEps = std::numeric_limits<glm::float32>::epsilon();

// https://iquilezles.org/articles/intersectors/
bool intersectSphere [[maybe_unused]] (const Ray & ray, const glm::vec3 & center, glm::float32 radius)
{
    glm::vec3 oc = center - ray.pos;
    glm::float32 l = glm::dot(ray.dir, oc);
    if (l < 0.0f) {
        return false;
    }
    glm::vec3 ll = ray.dir * l;
    return radius * radius > glm::dot(oc, oc) - glm::dot(ll, ll);
}

// TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
bool rayTriangleIntersectMoeller [[maybe_unused]] (const Ray & ray, const scene_data::Triangle & triangle, glm::vec3 & normal, glm::vec3 & uvw, glm::float32 & t)
{
    const glm::vec3 v1v0 = triangle.b - triangle.a;
    const glm::vec3 v2v0 = triangle.c - triangle.a;
    const glm::vec3 tVec = ray.pos - triangle.a;
    normal = glm::cross(v1v0, v2v0);
    const glm::float32 invDenom = 1.0f / glm::dot(ray.dir, normal);
    t = -glm::dot(normal, tVec) * invDenom;
    if (t <= 0.0f) {
        return false;
    }
    const glm::vec3 q = glm::cross(tVec, ray.dir);
    uvw.y = invDenom * glm::dot(-q, v2v0);
    uvw.z = invDenom * glm::dot(q, v1v0);
    uvw.x = 1.0f - uvw.y - uvw.z;
    return glm::all(glm::lessThanEqual(glm::vec3(-kEps), uvw));
}

glm::vec3 stableTriangleNormal(const glm::vec3 & a, const glm::vec3 & b, const glm::vec3 & c)
{
    const glm::vec3 ab{a.z * b.y, a.x * b.z, a.y * b.x};
    const glm::vec3 bc{b.z * c.y, b.x * c.z, b.y * c.x};
    const glm::vec3 AB{a.y * b.z - ab.x, a.z * b.x - ab.y, a.x * b.y - ab.z};
    const glm::vec3 BC{b.y * c.z - bc.x, b.z * c.x - bc.y, b.x * c.y - bc.z};
    return glm::mix(BC, AB, glm::lessThan(glm::abs(ab), glm::abs(bc)));
}

bool rayTriangleIntersectPluecker [[maybe_unused]] (const Ray & ray, const scene_data::Triangle triangle, glm::vec3 & uvw, glm::vec3 & normal, glm::float32 & t)
{
    const glm::vec3 a = triangle.a - ray.pos;
    const glm::vec3 b = triangle.b - ray.pos;
    const glm::vec3 c = triangle.c - ray.pos;

    const glm::vec3 x = c - b;
    const glm::vec3 y = a - c;
    const glm::vec3 z = b - a;

    uvw = glm::vec3(glm::dot(glm::cross(x, b + c), ray.dir), glm::dot(glm::cross(y, c + a), ray.dir), glm::dot(glm::cross(z, a + b), ray.dir));

    const glm::float32 sum = uvw.x + uvw.y + uvw.z;
    const glm::float32 eps = kUlp * glm::abs(sum);
    if (glm::any(glm::lessThan(glm::vec3(eps), uvw)) && glm::any(glm::lessThan(uvw, glm::vec3(eps)))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = glm::dot(normal, a) / glm::dot(ray.dir, normal);
    if (t <= 0.0f) {
        return false;
    }
    uvw = glm::min(uvw / sum, 1.0f);
    return true;
}

#if 0
#define rayTriangleIntersect rayTriangleIntersectPluecker
#else
#define rayTriangleIntersect rayTriangleIntersectMoeller
#endif

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

    [[nodiscard]] glm::uint findNode(glm::uint nodeIndex, const glm::vec3 & rayPos) const
    {
        while (nodeIndex != 0u) {
            const Node & node = nodes.at(nodeIndex);
            if (glm::all(glm::lessThanEqual(node.aabbMin, rayPos)) && glm::all(glm::lessThanEqual(rayPos, node.aabbMax))) {
                break;
            }
            nodeIndex = nodeParents.at(nodeIndex);
        }
        return nodeIndex;
    }

    [[nodiscard]] bool bruteForceRay(const Ray & ray, Hit & hit) const
    {
        bool isHit = false;
        for (const scene_data::Triangle & triangle : triangles) {
            glm::vec3 uvw;
            glm::vec3 normal;
            glm::float32 t;
            if (rayTriangleIntersect(ray, triangle, uvw, normal, t)) {
                if (t < hit.t) {
                    hit.triangle = utils::autoCast(std::distance(std::data(triangles), &triangle));
                    hit.uvw = uvw;
                    hit.normal = normal;
                    hit.t = t;
                }
                isHit = true;
            }
        }
        return isHit;
    }

    [[nodiscard]] bool traceRay(glm::uint nodeIndex, const Ray & ray, Hit & hit) const
    {
        const glm::vec3 invDir = 1.0f / ray.dir;
        const glm::bvec3 corner = glm::lessThan(invDir, glm::vec3{0.0f});
        const Node * node = &nodes.at(nodeIndex);
        glm::vec3 aabbHitT = (glm::mix(node->aabbMin, node->aabbMax, corner) - ray.pos) * invDir;
        glm::float32 t = glm::max(aabbHitT.x, glm::max(aabbHitT.y, aabbHitT.z));
        for (;;) {
            while (!(node->splitDimension < 0)) {
                if (corner[node->splitDimension] == ((node->splitPos - ray.pos[node->splitDimension]) * invDir[node->splitDimension] < t)) {
                    nodeIndex = node->leftChild;
                } else {
                    nodeIndex = node->rightChild;
                }
                node = &nodes.at(nodeIndex);
            }
            const glm::uint polygonStart = node->leftChild;
            const glm::uint polygonEnd = polygonStart + node->rightChild;
            for (glm::uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
                const glm::uint triangle = polygons.at(polygon);
                glm::vec3 uvw;
                glm::vec3 normal;
                if (rayTriangleIntersect(ray, triangles.at(triangle), uvw, normal, t)) {
                    if (t < hit.t) {
                        hit.triangle = triangle;
                        hit.uvw = uvw;
                        hit.normal = normal;
                        hit.t = t;
                    }
                }
            }
            aabbHitT = (glm::mix(node->aabbMax, node->aabbMin, corner) - ray.pos) * invDir;
            t = glm::min(aabbHitT.x, glm::min(aabbHitT.y, aabbHitT.z));
            if (hit.t <= t) {
                return true;
            }
            const glm::ivec3 indices = glm::mix(glm::ivec3(0), glm::ivec3(0, 1, 2), glm::equal(aabbHitT, glm::vec3(t)));
            const glm::int32 ropeDirection = glm::max(indices.x, glm::max(indices.y, indices.z));
            nodeIndex = corner[ropeDirection] ? node->leftRope[ropeDirection] : node->rightRope[ropeDirection];
            // SPDLOG_INFO("{} {}", __LINE__, nodeIndex);
            if (nodeIndex == 0u) {
                break;
            }
            node = &nodes.at(nodeIndex);
        }
        return hit.triangle != std::numeric_limits<glm::uint>::max();
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
    const glm::uint nodeIndex = impl_->findNode(kRootNodeIndex, ray.pos);
    for (gli::int32 y = 0; y < extent.y; ++y) {
        const glm::float32 locY = (utils::safeCast<glm::float32>(y) + 0.5f) * invExtent.y;
        const glm::vec3 left = glm::mix(leftBottom, leftTop, locY);
        const glm::vec3 right = glm::mix(rightBottom, rightTop, locY);
        for (gli::int32 x = 0; x < extent.x; ++x) {
            const glm::float32 locX = (utils::safeCast<glm::float32>(x) + 0.5f) * invExtent.x;
            ray.dir = glm::normalize(glm::mix(left, right, locX));
            glm::vec4 color;
            Hit hit;
            hit.triangle = std::numeric_limits<glm::uint>::max();
            hit.t = std::numeric_limits<glm::float32>::infinity();
            if (impl_->traceRay(nodeIndex, ray, hit)) {
                color = glm::vec4{hit.uvw, 1.0f};
            } else {
                color = impl_->clearColor;
            }
            constexpr auto kScale = static_cast<glm::vec4::value_type>(std::numeric_limits<PixelType::value_type>::max());
            target.store(gli::extent2d{x, extent.y - y - 1}, kLevel, PixelType(glm::clamp(color, glm::vec4{0.0f}, glm::vec4{1.0f}) * kScale));
        }
    }
}

}  // namespace soft_renderer
