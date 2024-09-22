#include <softrenderer/softrenderer.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <glm/ext/vector_bool3.hpp>
#include <glm/vec2.hpp>
#include <vulkan/vulkan.hpp>

#include <limits>
#include <string>

namespace softrenderer
{
namespace
{
constexpr glm::uint kRootNodeIndex = 0;

struct Ray
{
    glm::vec3 src;
    glm::vec3 dir;
};

struct Hit
{
    glm::uint triangle;
    glm::float32 t;
    glm::vec2 uv;
};

#pragma pack(push, 1)

struct Triangle
{
    glm::vec3 a, b, c;
};
static_assert(std::is_standard_layout_v<Triangle>);
static_assert(sizeof(Triangle) == 36);

struct Node
{
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    glm::uvec3 leftRope;
    glm::uvec3 rightRope;
    glm::int32 splitDimension;
    glm::float32 splitPos;
    glm::uint leftChild;
    glm::uint rightChild;
};
static_assert(std::is_standard_layout_v<Triangle>);
static_assert(sizeof(Node) == 64);

#pragma pack(pop)

bool rayTriangleIntersect(const Ray & ray, Hit & hit, const Triangle & triangle, glm::float32 tNear, glm::float32 tFar)
{
    // TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
    const glm::vec3 v1v0 = triangle.b - triangle.a;
    const glm::vec3 v2v0 = triangle.c - triangle.a;
    const glm::vec3 rov0 = ray.src - triangle.a;
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

}  // namespace

struct SoftRenderer::Impl
{
    const std::string name;
    const glm::vec4 clearColor;

    builder::TreePtr tree;

    std::vector<Triangle> triangles;
    std::vector<glm::uint> polygons;
    std::vector<Node> nodes;

    explicit Impl(std::string_view name, glm::vec4 clearColor)
        : name{name}
        , clearColor{clearColor}
    {}

    [[nodiscard]] glm::uint findNode(glm::uint nodeIndex, const Ray & ray)
    {
        for (;;) {
            const Node & node = nodes.at(nodeIndex);
            const glm::int32 splitDimension = node.splitDimension;
            if (splitDimension < 0) {
                return nodeIndex;
            }
            if (ray.src[splitDimension] < node.splitPos) {
                nodeIndex = node.leftChild;
            } else {
                nodeIndex = node.rightChild;
            }
        }
    }

    [[nodiscard]] bool traceRay(glm::uint nodeIndex, const Ray & ray, Hit & hit)
    {
        const glm::vec3 invDir = 1.0f / ray.dir;
        const glm::bvec3 corner = glm::lessThan(invDir, glm::vec3(0.0f));
        glm::vec3 aabbHitT = (glm::mix(nodes.at(nodeIndex).aabbMin, nodes.at(nodeIndex).aabbMax, corner) - ray.src) * invDir;
        glm::float32 tMin = glm::max(aabbHitT.x, glm::max(aabbHitT.y, aabbHitT.z));
        do {
            const glm::float32 tNear = glm::max(0.0f, tMin);
            for (;;) {
                const Node & node = nodes.at(nodeIndex);
                const glm::int32 splitDimension = node.splitDimension;
                if (splitDimension < 0) {
                    break;
                }
                if (corner[splitDimension] == ((node.splitPos - ray.src[splitDimension]) * invDir[splitDimension] < tNear)) {
                    nodeIndex = node.leftChild;
                } else {
                    nodeIndex = node.rightChild;
                }
            }
            const Node & node = nodes.at(nodeIndex);
            aabbHitT = (glm::mix(node.aabbMax, node.aabbMin, corner) - ray.src) * invDir;
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
                if (rayTriangleIntersect(ray, hit, triangles.at(closerHit.triangle), tNear, tFar)) {
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

SoftRenderer::SoftRenderer(std::string_view name, glm::vec4 clearColor)
    : impl_{std::make_unique<Impl>(name, clearColor)}
{}

void SoftRenderer::unsetTree()
{
    ASSERT(impl_->tree);
    impl_->tree.reset();
}

void SoftRenderer::updateTree(const builder::TreePtr & tree)
{
    ASSERT(!impl_->tree);
    impl_->tree = tree;
}

const builder::TreePtr & SoftRenderer::getTree() const &
{
    return impl_->tree;
}

void SoftRenderer::render(const FrameSettings & frameSettings, gli::texture2d & target) const
{
    INVARIANT(!target.empty(), "");
    INVARIANT(target.format() == gli::FORMAT_RGB32_SFLOAT_PACK32, "{}", fmt::underlying(target.format()));
    constexpr size_t kLevel = 0;
    target.clear(impl_->clearColor);
    const gli::extent2d extent = target.extent();
    const glm::float32 dy = glm::tan(frameSettings.fov * 0.5f);
    const glm::float32 dx = dy * (utils::safeCast<glm::float32>(extent.x) / utils::safeCast<glm::float32>(extent.y));
    const glm::vec3 leftTop = glm::rotate(frameSettings.orientation, glm::vec3{-dx, dy, 1.0f});
    const glm::vec3 rightTop = glm::rotate(frameSettings.orientation, glm::vec3{dx, dy, 1.0f});
    const glm::vec3 leftBottom = glm::rotate(frameSettings.orientation, glm::vec3{-dx, -dy, 1.0f});
    const glm::vec3 rightBottom = glm::rotate(frameSettings.orientation, glm::vec3{dx, -dy, 1.0f});
    const glm::vec2 invExtent = 1.0f / glm::vec2(extent);
    Ray ray;
    ray.src = frameSettings.position;
    Hit hit;
    hit.t = std::numeric_limits<glm::float32>::infinity();
    const glm::uint nodeIndex = impl_->findNode(kRootNodeIndex, ray);
    for (gli::int32 y = 0; y < extent.y; ++y) {
        const glm::float32 locY = (utils::safeCast<glm::float32>(y) + 0.5f) * invExtent.y;
        const glm::vec3 left = glm::mix(leftBottom, leftTop, locY);
        const glm::vec3 right = glm::mix(rightBottom, rightTop, locY);
        for (gli::int32 x = 0; x < extent.x; ++x) {
            const glm::float32 locX = (utils::safeCast<glm::float32>(x) + 0.5f) * invExtent.x;
            ray.dir = glm::normalize(glm::mix(left, right, locX));
            glm::vec4 texel;
            if (impl_->traceRay(nodeIndex, ray, hit)) {
                texel = glm::vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv, 1.0f);
            } else {
                texel = impl_->clearColor;
            }
            target.store(gli::extent2d{x, y}, kLevel, texel);
        }
    }
}

}  // namespace softrenderer
