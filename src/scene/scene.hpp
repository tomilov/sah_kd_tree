#pragma once

#include <scene/fwd.hpp>
#include <utils/mem_array.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <limits>
#include <type_traits>
#include <vector>

#include <cstddef>

#include <scene/scene_export.h>

namespace scene
{

using Position = glm::vec3;
static_assert(std::is_standard_layout_v<Position>);

#pragma pack(push, 1)

struct Triangle
{
    Position a, b, c;
};

#pragma pack(pop)
static_assert(std::is_standard_layout_v<Triangle>);

#pragma pack(push, 1)

struct VertexAttributes
{
    Position position;
};

#pragma pack(pop)
static_assert(std::is_standard_layout_v<VertexAttributes>);

#pragma pack(push, 1)

struct AABB
{
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};
};

#pragma pack(pop)
static_assert(std::is_standard_layout_v<AABB>);

struct SCENE_EXPORT Node
{
    size_t parent = 0;  // index in Scene::Nodes
    glm::mat4 transform{1.0f};
    std::vector<size_t> meshes = {};    // indices in Scene::meshes
    std::vector<size_t> children = {};  // indices in Scene::nodes
    AABB aabb;
};

struct SCENE_EXPORT Mesh
{
    uint32_t indexOffset = 0, indexCount = 0;    // range in Scene::indices
    uint32_t vertexOffset = 0, vertexCount = 0;  // range in Scene::vertices
    AABB aabb;
};
static_assert(std::is_standard_layout_v<Mesh>);

struct SCENE_EXPORT Scene
{
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    AABB aabb;

    utils::MemArray<uint32_t> indices;
    utils::MemArray<VertexAttributes> vertices;

    [[nodiscard]] size_t instanceCount(size_t rootNodeIndex = 0) const;

    void updateAABBs();

    [[nodiscard]] utils::MemArray<Triangle> makeTriangles() const;
    [[nodiscard]] utils::MemArray<Triangle> makeTriangles(size_t rootNodeIndex) const;
};
static_assert(std::is_nothrow_move_constructible_v<Scene>);

}  // namespace scene
