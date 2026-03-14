#pragma once

#include <builder/fwd.hpp>
#include <soft_renderer/fwd.hpp>
#include <utils/mem_array.hpp>

#include <glm/fwd.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <type_traits>

namespace soft_renderer
{

inline constexpr glm::uint kRootNodeIndex = 0;

struct Ray
{
    glm::vec3 pos;
    glm::vec3 dir;
};

struct Hit
{
    glm::uint triangle;
    glm::vec3 uvw;
    glm::vec3 normal;
    glm::float32 t;
};

#pragma pack(push, 1)

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
static_assert(std::is_standard_layout_v<Node>);
static_assert(std::is_trivially_copyable_v<Node>);
static_assert(sizeof(Node) == 64);

#pragma pack(pop)

void importTree(
    builder::Tree tree,
    utils::MemArray<glm::uvec3> & indices,
    utils::MemArray<glm::vec3> & vertices,
    utils::MemArray<glm::uint> & polygons,
    utils::MemArray<Node> & nodes,
    utils::MemArray<glm::uint> & nodeParents);

}  // namespace soft_renderer
