#include <builder/builder.hpp>
#include <scene_data/scene_data.hpp>
#include <soft_renderer/soft_renderer.hpp>
#include <soft_renderer/tree.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/mem_array.hpp>

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
#include <ranges>
#include <string>
#include <string_view>
#include <thread>

#include <cmath>

#include <cuda.h>
#include <omp.h>

namespace soft_renderer
{
namespace
{

const glm::float32 kUlp = std::nextafter(0.0f, 1.0f);
constexpr glm::float32 kEps = std::numeric_limits<glm::float32>::epsilon();

// https://iquilezles.org/articles/intersectors/
bool intersectSphere [[maybe_unused]] (
    const Ray & ray,
    const glm::vec3 & center,
    glm::float32 radius)
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
bool rayTriangleIntersectMoeller [[maybe_unused]] (
    const Ray & ray,
    const scene_data::Triangle & triangle,
    glm::vec3 & uvw,
    glm::vec3 & normal,
    glm::float32 & t)
{
    const glm::vec3 ca = triangle.a - triangle.c;
    const glm::vec3 bc = triangle.c - triangle.b;
    const glm::vec3 c = ray.pos - triangle.c;
    normal = glm::cross(bc, ca);
    const glm::float32 invPlaneDist = 1.0f / glm::dot(ray.dir, normal);
    t = -glm::dot(normal, c) * invPlaneDist;
    if (t <= 0.0f) {
        return false;
    }
    const glm::vec3 q = glm::cross(c, ray.dir);
    uvw.x = glm::dot(q, bc) * invPlaneDist;
    uvw.y = glm::dot(q, ca) * invPlaneDist;
    uvw.z = 1.0f - (uvw.x + uvw.y);
    return glm::all(glm::lessThanEqual(glm::vec3{-kEps}, uvw));
}

glm::vec3 stableTriangleNormal(
    const glm::vec3 & a,
    const glm::vec3 & b,
    const glm::vec3 & c)
{
    const glm::vec3 ab{a.z * b.y, a.x * b.z, a.y * b.x};
    const glm::vec3 bc{b.z * c.y, b.x * c.z, b.y * c.x};
    const glm::vec3 AB{(a.y * b.z) - ab.x, (a.z * b.x) - ab.y, (a.x * b.y) - ab.z};
    const glm::vec3 BC{(b.y * c.z) - bc.x, (b.z * c.x) - bc.y, (b.x * c.y) - bc.z};
    return glm::mix(BC, AB, glm::lessThan(glm::abs(ab), glm::abs(bc)));
}

bool rayTriangleIntersectPluecker [[maybe_unused]] (
    const Ray & ray,
    const scene_data::Triangle & triangle,
    glm::vec3 & uvw,
    glm::vec3 & normal,
    glm::float32 & t)
{
    const glm::vec3 a = triangle.a - ray.pos;
    const glm::vec3 b = triangle.b - ray.pos;
    const glm::vec3 c = triangle.c - ray.pos;

    const glm::vec3 x = c - b;
    const glm::vec3 y = a - c;
    const glm::vec3 z = b - a;

    uvw.x = glm::dot(glm::cross(x, b + c), ray.dir);
    uvw.y = glm::dot(glm::cross(y, c + a), ray.dir);
    uvw.z = glm::dot(glm::cross(z, a + b), ray.dir);

    const glm::float32 sum = uvw.x + uvw.y + uvw.z;
    const glm::float32 eps = kUlp * glm::abs(sum);
    if (glm::any(glm::lessThan(uvw, glm::vec3(-eps))) && glm::any(glm::lessThan(glm::vec3{eps}, uvw))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = glm::dot(a, normal) / glm::dot(ray.dir, normal);
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

    utils::MemArray<glm::uvec3> indices;
    utils::MemArray<glm::vec3> vertices;
    utils::MemArray<glm::uint> polygons;
    utils::MemArray<Node> nodes;
    utils::MemArray<glm::uint> nodeParents;

    Impl(
        std::string_view nameIn,
        const glm::vec4 & clearColorIn)
        : name{nameIn}
        , clearColor{clearColorIn}
    {
        omp_set_num_threads(utils::autoCast(std::thread::hardware_concurrency()));
    }

    [[nodiscard]] glm::uint findNode(
        glm::uint nodeIndex,
        const glm::vec3 & pos) const
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

    [[nodiscard]] scene_data::Triangle getTriangle(glm::uint t) const
    {
        const glm::uvec3 index = indices.at(t);
        return {
            .a = vertices.at(index.x),
            .b = vertices.at(index.y),
            .c = vertices.at(index.z),
        };
    }

    void traceRay(
        glm::uint nodeIndex,
        const Ray & ray,
        Hit & hit,
        glm::float32 tMin) const
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
                glm::vec3 uvw;
                glm::vec3 normal;
                if (rayTriangleIntersect(ray, getTriangle(triangle), uvw, normal, tMin)) {
                    if (tMin < hit.t) {
                        hit.triangle = triangle;
                        hit.uvw = uvw;
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
            const glm::ivec2 exitAxis = glm::mix(glm::ivec2(0), glm::ivec2(1, 2), glm::equal(glm::vec2{aabbHitT.y, aabbHitT.z}, glm::vec2{tMin}));
            const glm::int32 ropeDirection = glm::max(exitAxis.x, exitAxis.y);
            nodeIndex = corner[ropeDirection] ? node->leftRope[ropeDirection] : node->rightRope[ropeDirection];
        } while (nodeIndex != 0u);
    }
};

SoftRenderer::SoftRenderer(
    std::string_view name,
    const glm::vec4 & clearColor)
    : impl_{std::make_unique<Impl>(
          name,
          clearColor)}
{}

SoftRenderer::~SoftRenderer() = default;

SoftRenderer::SoftRenderer(SoftRenderer &&) noexcept = default;

void SoftRenderer::setTree(builder::Tree && builderTree)
{
    importTree(std::move(builderTree), impl_->indices, impl_->vertices, impl_->polygons, impl_->nodes, impl_->nodeParents);
}

bool SoftRenderer::hasTree() const
{
    if (impl_->indices.isEmpty()) {
        SKT_ASSERT(impl_->vertices.isEmpty());
        SKT_ASSERT(impl_->polygons.isEmpty());
        SKT_ASSERT(impl_->nodes.isEmpty());
        SKT_ASSERT(impl_->nodeParents.isEmpty());
        return false;
    }
    SKT_ASSERT(!impl_->vertices.isEmpty());
    SKT_ASSERT(!impl_->polygons.isEmpty());
    SKT_ASSERT(!impl_->nodes.isEmpty());
    SKT_ASSERT(!impl_->nodeParents.isEmpty());
    return true;
}

void SoftRenderer::render(
    const FrameSettings & frameSettings,
    gli::texture2d & target) const
{
    SKT_INVARIANT(!target.empty(), "");
    SKT_INVARIANT(target.format() == kTargetFormat, "{}", fmt::underlying(target.format()));
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
    const glm::uint nodeIndex = 0u;  // impl_->findNode(kRootNodeIndex, frameSettings.position);
#pragma omp parallel for schedule(dynamic, 1)
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
            hit.t = std::numeric_limits<glm::float32>::max();
            hit.uvw = {};
            impl_->traceRay(nodeIndex, ray, hit, tNear);
            glm::vec4 color;
            if (hit.triangle != std::numeric_limits<glm::uint>::max()) {
                color = glm::vec4{hit.uvw, impl_->clearColor.a};
            } else {
                color = impl_->clearColor;
            }
            constexpr glm::vec4::value_type kScale = utils::autoCast(std::numeric_limits<PixelType::value_type>::max());
            target.store(gli::extent2d{x, extent.y - y - 1}, kLevel, PixelType(kScale * glm::clamp(color, glm::vec4{0.0f}, glm::vec4{1.0f})));
        }
    }
}

}  // namespace soft_renderer
