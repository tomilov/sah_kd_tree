#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <memory>
#include <vector>

#include <cstddef>

#include <scene/scene_export.h>

namespace scene
{

using Position = glm::vec3;

struct Triangle
{
    Position a, b, c;
};

struct SCENE_EXPORT Triangles
{
    size_t triangleCount = 0;
    std::unique_ptr<Triangle[]> triangles;

    void resize(size_t newTraingleCount);
};

struct VertexAttributes
{
    Position position;
};

struct SCENE_EXPORT Node
{
    size_t parent = 0;  // index in Scene::Nodes
    glm::mat4 transform{1.0f};
    std::vector<size_t> meshes;    // indices in Scene::meshes
    std::vector<size_t> children;  // indices in Scene::nodes
};

struct SCENE_EXPORT Mesh
{
    uint32_t indexOffset = 0, indexCount = 0;    // indices in Scene::indices
    uint32_t vertexOffset = 0, vertexCount = 0;  // indices in Scene::vertices
};

struct SCENE_EXPORT Scene
{
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;

    size_t indexCount = 0;
    std::unique_ptr<uint32_t[]> indices;

    void resizeIndices(size_t newIndexCount);

    size_t vertexCount = 0;
    std::unique_ptr<VertexAttributes[]> vertices;

    void resizeVertices(size_t newVertexCount);

    size_t instanceCount(size_t rootNodeIndex = 0) const;

    Triangles makeTriangles() const;
    Triangles makeTriangles(size_t rootNodeIndex) const;
};

}  // namespace scene
