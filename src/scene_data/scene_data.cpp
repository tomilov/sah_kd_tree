#include <scene_data/fwd.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/mem_array.hpp>

#include <glm/common.hpp>

#include <iterator>

namespace scene_data
{

size_t SceneData::instanceCount(size_t rootNodeIndex) const
{
    size_t instanceCount = 0;
    const auto countInstances = [this, &instanceCount](const auto & self, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        instanceCount += node.meshes.size();
        for (size_t childIndex : node.children) {
            self(self, childIndex);
        }
    };
    countInstances(countInstances, rootNodeIndex);
    return instanceCount;
}

void SceneData::updateAABBs()
{
    for (Node & node : nodes) {
        for (size_t meshIndex : node.meshes) {
            const auto & mesh = meshes.at(meshIndex);
            node.aabb.min = glm::min(node.aabb.min, mesh.aabb.min);
            node.aabb.max = glm::max(node.aabb.max, mesh.aabb.max);
        }
        aabb.min = glm::min(aabb.min, node.aabb.min);
        aabb.max = glm::max(aabb.max, node.aabb.max);
    }
}

utils::MemArray<Triangle> SceneData::makeTriangles() const
{
    size_t vertexCount = 0;
    for (const Mesh & mesh : meshes) {
        INVARIANT((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
        vertexCount += mesh.indexCount;
    }

    utils::MemArray<Triangle> triangles{vertexCount / 3};
    auto * t = triangles.begin();
    const auto * v = vertices.begin();
    for (const Mesh & mesh : meshes) {
        const auto * index = indices.begin();
        std::advance(index, mesh.indexOffset);
        const auto * const endIndex = std::next(index, mesh.indexCount);
        while (index != endIndex) {
            INVARIANT(t < triangles.end(), "");
            uint32_t a = *index++;
            INVARIANT(a < mesh.vertexCount, "");
            uint32_t b = *index++;
            INVARIANT(b < mesh.vertexCount, "");
            uint32_t c = *index++;
            INVARIANT(c < mesh.vertexCount, "");
            *t++ = {
                .a = v[mesh.vertexOffset + a].position,
                .b = v[mesh.vertexOffset + b].position,
                .c = v[mesh.vertexOffset + c].position,
            };
        }
    }
    return triangles;
}

utils::MemArray<Triangle> SceneData::makeTriangles(size_t rootNodeIndex) const
{
    size_t vertexCount = 0;
    const auto countTriangles = [this, &vertexCount](const auto & self, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            INVARIANT((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
            vertexCount += mesh.indexCount;
        }
        for (size_t childIndex : node.children) {
            self(self, childIndex);
        }
    };
    countTriangles(countTriangles, rootNodeIndex);

    utils::MemArray<Triangle> triangles{vertexCount / 3};
    auto * t = triangles.begin();
    const auto * v = vertices.begin();
    const auto traverseNodes = [this, &t, &triangles, v](const auto & self, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            const auto * index = indices.begin();
            std::advance(index, mesh.indexOffset);
            const auto * const endIndex = std::next(index, mesh.indexCount);
            while (index != endIndex) {
                INVARIANT(t < triangles.end(), "");
                uint32_t a = *index++;
                INVARIANT(a < mesh.vertexCount, "");
                uint32_t b = *index++;
                INVARIANT(b < mesh.vertexCount, "");
                uint32_t c = *index++;
                INVARIANT(c < mesh.vertexCount, "");
                *t++ = {
                    .a = v[mesh.vertexOffset + a].position,
                    .b = v[mesh.vertexOffset + b].position,
                    .c = v[mesh.vertexOffset + c].position,
                };
            }
        }
        for (size_t childIndex : node.children) {
            self(self, childIndex);
        }
    };
    traverseNodes(traverseNodes, rootNodeIndex);
    ASSERT(t == triangles.end());
    return triangles;
}

}  // namespace scene_data
