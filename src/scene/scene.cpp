#include <scene/scene.hpp>
#include <utils/assert.hpp>

namespace scene
{

void Triangles::resize(size_t newTraingleCount)
{
    triangleCount = newTraingleCount;
    triangles = std::make_unique<Triangle[]>(newTraingleCount);
}

void Scene::resizeIndices(size_t newIndexCount)
{
    indexCount = newIndexCount;
    indices = std::make_unique<uint32_t[]>(newIndexCount);
}

void Scene::resizeVertices(size_t newVertexCount)
{
    vertexCount = newVertexCount;
    vertices = std::make_unique<VertexAttributes[]>(newVertexCount);
}

size_t Scene::instanceCount(size_t rootNodeIndex) const
{
    size_t instanceCount = 0;
    const auto countInstances = [this, &instanceCount](const auto & countInstances, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        instanceCount += node.meshes.size();
        for (size_t childIndex : node.children) {
            countInstances(countInstances, childIndex);
        }
    };
    countInstances(countInstances, rootNodeIndex);
    return instanceCount;
}

Triangles Scene::makeTriangles() const
{
    size_t vertexCount = 0;
    for (const Mesh & mesh : meshes) {
        INVARIANT((mesh.indexCount % 3) == 0, "");
        vertexCount += mesh.indexCount;
    }

    Triangles triangles;
    triangles.resize(vertexCount / 3);
    auto t = triangles.triangles.get();
    const auto tEnd = std::next(t, triangles.triangleCount);
    for (const Mesh & mesh : meshes) {
        auto index = indices.get();
        std::advance(index, mesh.indexOffset);
        auto endIndex = std::next(index, mesh.indexCount);
        while (index != endIndex) {
            INVARIANT(t < tEnd, "");
            uint32_t a = *index++;
            INVARIANT(a < mesh.vertexCount, "");
            uint32_t b = *index++;
            INVARIANT(b < mesh.vertexCount, "");
            uint32_t c = *index++;
            INVARIANT(c < mesh.vertexCount, "");
            *t++ = {
                .a = vertices[mesh.vertexOffset + a].position,
                .b = vertices[mesh.vertexOffset + b].position,
                .c = vertices[mesh.vertexOffset + c].position,
            };
        }
    }
    return triangles;
}

Triangles Scene::makeTriangles(size_t rootNodeIndex) const
{
    size_t vertexCount = 0;
    const auto countTriangles = [this, &vertexCount](const auto & countTriangles, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            INVARIANT((mesh.indexCount % 3) == 0, "");
            vertexCount += mesh.indexCount;
        }
        for (size_t childIndex : node.children) {
            countTriangles(countTriangles, childIndex);
        }
    };
    countTriangles(countTriangles, rootNodeIndex);

    Triangles triangles;
    triangles.resize(vertexCount / 3);
    auto t = triangles.triangles.get();
    const auto tEnd = std::next(t, triangles.triangleCount);
    const auto traverseNodes = [this, &t, &tEnd](const auto & traverseNodes, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            auto index = indices.get();
            std::advance(index, mesh.indexOffset);
            auto endIndex = std::next(index, mesh.indexCount);
            while (index != endIndex) {
                INVARIANT(t < tEnd, "");
                uint32_t a = *index++;
                INVARIANT(a < mesh.vertexCount, "");
                uint32_t b = *index++;
                INVARIANT(b < mesh.vertexCount, "");
                uint32_t c = *index++;
                INVARIANT(c < mesh.vertexCount, "");
                *t++ = {
                    .a = vertices[mesh.vertexOffset + a].position,
                    .b = vertices[mesh.vertexOffset + b].position,
                    .c = vertices[mesh.vertexOffset + c].position,
                };
            }
        }
        for (size_t childIndex : node.children) {
            traverseNodes(traverseNodes, childIndex);
        }
    };
    traverseNodes(traverseNodes, rootNodeIndex);
    ASSERT(t == tEnd);
    return triangles;
}

}  // namespace scene
