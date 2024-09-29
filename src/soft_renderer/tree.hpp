#pragma once

#include <builder/fwd.hpp>
#include <scene_data/fwd.hpp>
#include <soft_renderer/fwd.hpp>

#include <glm/fwd.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <type_traits>
#include <vector>

namespace soft_renderer
{

inline constexpr glm::uint kRootNodeIndex = 0;

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
static_assert(sizeof(Node) == 64);

#pragma pack(pop)

void importTree(builder::Tree tree, std::vector<scene_data::Triangle> & triangles, std::vector<glm::uint> & polygons, std::vector<Node> & nodes, std::vector<glm::uint> & nodeParents);

}  // namespace soft_renderer
