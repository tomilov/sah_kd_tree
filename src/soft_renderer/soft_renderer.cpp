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
#include <spdlog/spdlog.h>

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
bool rayTriangleIntersectMoeller [[maybe_unused]] (const Ray & ray, const scene_data::Triangle & triangle, glm::vec2 & uv, glm::vec3 & normal, glm::float32 & t)
{
    const glm::vec3 ca = triangle.b - triangle.a;
    const glm::vec3 bc = triangle.c - triangle.a;
    const glm::vec3 c = ray.pos - triangle.a;
    normal = glm::cross(bc, ca);
    const glm::float32 invPlaneDist = 1.0f / glm::dot(ray.dir, normal);
    t = -glm::dot(normal, c) * invPlaneDist;
    if (t <= 0.0f) {
        return false;
    }
    const glm::vec3 q = glm::cross(c, ray.dir);
    uv.x = glm::dot(-q, bc) * invPlaneDist;
    uv.y = glm::dot(q, ca) * invPlaneDist;
    return glm::all(glm::lessThanEqual(glm::vec3{-kEps}, glm::vec3{uv, 1.0f - uv.x - uv.y}));
}

glm::vec3 stableTriangleNormal(const glm::vec3 & a, const glm::vec3 & b, const glm::vec3 & c)
{
    const glm::vec3 ab{a.z * b.y, a.x * b.z, a.y * b.x};
    const glm::vec3 bc{b.z * c.y, b.x * c.z, b.y * c.x};
    const glm::vec3 AB{a.y * b.z - ab.x, a.z * b.x - ab.y, a.x * b.y - ab.z};
    const glm::vec3 BC{b.y * c.z - bc.x, b.z * c.x - bc.y, b.x * c.y - bc.z};
    return glm::mix(BC, AB, glm::lessThan(glm::abs(ab), glm::abs(bc)));
}

bool rayTriangleIntersectPluecker [[maybe_unused]] (const Ray & ray, const scene_data::Triangle triangle, glm::vec2 & uv, glm::vec3 & normal, glm::float32 & t)
{
    const glm::vec3 a = triangle.a - ray.pos;
    const glm::vec3 b = triangle.b - ray.pos;
    const glm::vec3 c = triangle.c - ray.pos;

    const glm::vec3 x = c - b;
    const glm::vec3 y = a - c;
    const glm::vec3 z = b - a;

    uv.x = glm::dot(glm::cross(x, b + c), ray.dir);
    uv.y = glm::dot(glm::cross(y, c + a), ray.dir);
    const glm::float32 w = glm::dot(glm::cross(z, a + b), ray.dir);

    const glm::float32 sum = uv.x + uv.y + w;
    const glm::float32 eps = kUlp * glm::abs(sum);
    if (glm::any(glm::lessThan(glm::vec3{uv, w}, glm::vec3(-kEps))) && glm::any(glm::lessThan(glm::vec3{eps}, glm::vec3{uv, w}))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = glm::dot(a, normal) / glm::dot(ray.dir, normal);
    if (t <= 0.0f) {
        return false;
    }
    uv = glm::min(uv / sum, 1.0f);
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

    [[nodiscard]] glm::uint findNode(glm::uint nodeIndex, const glm::vec3 & pos) const
    {
        while (nodeIndex != 0u) {
            const Node & node = nodes.at(nodeIndex);
            if (glm::all(glm::lessThanEqual(node.aabbMin, pos)) && glm::all(glm::lessThanEqual(pos, node.aabbMax))) {
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
            glm::vec2 uv;
            glm::vec3 normal;
            glm::float32 tMin;
            if (rayTriangleIntersect(ray, triangle, uv, normal, tMin)) {
                if (tMin < hit.t) {
                    hit.triangle = utils::autoCast(std::distance(std::data(triangles), &triangle));
                    hit.uv = uv;
                    hit.normal = normal;
                    hit.t = tMin;
                }
                isHit = true;
            }
        }
        return isHit;
    }

    void traceRay(glm::uint nodeIndex, const Ray & ray, Hit & hit, glm::float32 tMin) const
    {
        const glm::vec3 invDir = 1.0f / ray.dir;
        const glm::bvec3 corner = glm::lessThan(invDir, glm::vec3{0.0f});
        do {
            const Node * node = &nodes.at(nodeIndex);
            while (!(node->splitDimension < 0)) {
                if (corner[node->splitDimension] == ((node->splitPos - ray.pos[node->splitDimension]) * invDir[node->splitDimension] < tMin)) {
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
                glm::vec2 uv;
                glm::vec3 normal;
                if (rayTriangleIntersect(ray, triangles.at(triangle), uv, normal, tMin)) {
                    if (tMin < hit.t) {
                        hit.triangle = triangle;
                        hit.uv = uv;
                        hit.normal = normal;
                        hit.t = tMin;
                    }
                }
            }
            const glm::vec3 aabbHitT = (glm::mix(node->aabbMax, node->aabbMin, corner) - ray.pos) * invDir;
            tMin = glm::min(aabbHitT.x, glm::min(aabbHitT.y, aabbHitT.z));
            if (hit.t <= tMin) {
                break;
            }
            const glm::ivec2 indices = glm::mix(glm::ivec2(0), glm::ivec2(1, 2), glm::equal(glm::vec2{aabbHitT.y, aabbHitT.z}, glm::vec2{tMin}));
            const glm::int32 ropeDirection = glm::max(indices.x, indices.y);
            nodeIndex = corner[ropeDirection] ? node->leftRope[ropeDirection] : node->rightRope[ropeDirection];
        } while (nodeIndex != 0u);
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
    const glm::float32 tNear = 0.0f;
    const glm::uint nodeIndex = 0u;  // impl_->findNode(kRootNodeIndex, ray.pos);
    for (gli::int32 y = 0; y < extent.y; ++y) {
        const glm::float32 locY = (utils::safeCast<glm::float32>(y) + 0.5f) * invExtent.y;
        const glm::vec3 left = glm::mix(leftBottom, leftTop, locY);
        const glm::vec3 right = glm::mix(rightBottom, rightTop, locY);
        for (gli::int32 x = 0; x < extent.x; ++x) {
            const glm::float32 locX = (utils::safeCast<glm::float32>(x) + 0.5f) * invExtent.x;
            Ray ray;
            ray.pos = frameSettings.position;
            ray.dir = glm::normalize(glm::mix(left, right, locX));
            Hit hit;
            hit.triangle = std::numeric_limits<glm::uint>::max();
            hit.t = std::numeric_limits<glm::float32>::infinity();
            impl_->traceRay(nodeIndex, ray, hit, tNear);
            glm::vec4 color;
            if (hit.triangle != std::numeric_limits<glm::uint>::max()) {
                color = glm::vec4{hit.uv, 1.0f - hit.uv.x - hit.uv.y, 1.0f};
            } else {
                color = impl_->clearColor;
            }
            constexpr auto kScale = static_cast<glm::vec4::value_type>(std::numeric_limits<PixelType::value_type>::max());
            target.store(gli::extent2d{x, extent.y - y - 1}, kLevel, PixelType(glm::clamp(color, glm::vec4{0.0f}, glm::vec4{1.0f}) * kScale));
        }
    }
}

}  // namespace soft_renderer
