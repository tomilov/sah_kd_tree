#pragma once

#include <scene_data/fwd.hpp>
#include <utils/mem_array.hpp>
#include <utils/noncopyable.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <scene_data/scene_data_export.h>

namespace scene_data
{

using Position = glm::vec3;
static_assert(std::is_standard_layout_v<Position>);
static_assert(std::is_trivially_copyable_v<Position>);

using Index = glm::uint;

#pragma pack(push, 1)

struct Triangle
{
    Position a, b, c;
};
static_assert(std::is_standard_layout_v<Triangle>);
static_assert(std::is_trivially_copyable_v<Triangle>);

struct VertexAttributes
{
    Position position;
};
static_assert(std::is_standard_layout_v<VertexAttributes>);
static_assert(std::is_trivially_copyable_v<VertexAttributes>);

struct AABB
{
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};
};
static_assert(std::is_standard_layout_v<AABB>);
static_assert(std::is_trivially_copyable_v<AABB>);

#pragma pack(pop)

struct SCENE_DATA_EXPORT Node
{
    size_t parent = 0;  // index in scene_data::nodes
    glm::mat4 transform{1.0f};
    std::vector<size_t> meshes;    // indices in Scene::meshes
    std::vector<size_t> children;  // indices in Scene::nodes
    AABB aabb = {};
};

struct SCENE_DATA_EXPORT Mesh
{
    // TODO(tomilov): make SceneData chunked, change uint32_t to size_t (uint32_t is enough for (1.5 * vertex + 3 * index) * 4G = 120GB scene)
    size_t indexOffset = 0;
    uint32_t indexCount = 0;  // range in Scene::indices
    size_t vertexOffset = 0;
    uint32_t vertexCount = 0;  // range in Scene::vertices
    AABB aabb = {};
};

struct SCENE_DATA_EXPORT SceneData : utils::OneTime<SceneData>
{
    std::string name;

    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    AABB aabb = {};

    utils::MemArray<Index> indices;
    utils::MemArray<VertexAttributes> vertices;

    [[nodiscard]] size_t instanceCount(size_t rootNodeIndex = 0) const;

    void updateAABBs();

    void collectScene(
        utils::MemArray<Index> & indices,
        utils::MemArray<Position> & vertices) const;
    void collectScene(
        size_t rootNodeIndex,
        utils::MemArray<Index> & indices,
        utils::MemArray<Position> & vertices) const;
};

}  // namespace scene_data
