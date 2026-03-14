#include <common/config.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/mem_array.hpp>

#include <glm/common.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <span>

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

void SceneData::collectScene(
    utils::MemArray<Index> & outIndices,
    utils::MemArray<Position> & outVertices) const
{
    if constexpr (sah_kd_tree::kIsDebugBuild) {
        size_t indexCount = 0;
        size_t vertexCount = 0;
        for (const Mesh & mesh : meshes) {
            ASSERT_MSG((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
            ASSERT(indexCount == mesh.indexOffset);
            ASSERT(vertexCount == mesh.vertexOffset);
            indexCount += mesh.indexCount;
            vertexCount += mesh.vertexCount;
        }
        ASSERT(!std::empty(meshes));
        ASSERT(indexCount == meshes.back().indexOffset + meshes.back().indexCount);
        ASSERT(vertexCount == meshes.back().vertexOffset + meshes.back().vertexCount);
        ASSERT(indexCount == indices.getCount());
        ASSERT(vertexCount == vertices.getCount());
    }
    outIndices.setCount(indices.getCount());

    outVertices.setCount(vertices.getCount());
    std::ranges::transform(vertices, outVertices.begin(), &VertexAttributes::position);

    std::span<const Index> inIndices{indices.cbegin(), indices.cend()};
    auto * outIndex = outIndices.begin();
    for (const Mesh & mesh : meshes) {
        const auto addVertexOffset = [&mesh, vertexCount = outVertices.getCount()](Index i) -> Index
        {
            ASSERT(i < mesh.vertexCount);
            ASSERT(mesh.vertexOffset + i < vertexCount);
            return mesh.vertexOffset + i;
        };
        outIndex = std::ranges::transform(inIndices.first(mesh.indexCount), outIndex, addVertexOffset).out;
        inIndices = inIndices.subspan(mesh.indexCount);
    }
    ASSERT(std::empty(inIndices));
    ASSERT(outIndex == outIndices.end());
}

void SceneData::collectScene(
    size_t rootNodeIndex,
    utils::MemArray<Index> & outIndices,
    utils::MemArray<Position> & outVertices) const
{
    size_t indexCount = 0;
    size_t vertexCount = 0;
    const auto countIndicesAndVertices = [this, &indexCount, &vertexCount](const auto & self, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            INVARIANT((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
            indexCount += mesh.indexCount;
            vertexCount += mesh.vertexCount;
        }
        for (size_t childIndex : node.children) {
            self(self, childIndex);
        }
    };
    countIndicesAndVertices(countIndicesAndVertices, rootNodeIndex);
    outIndices.setCount(indexCount);
    outVertices.setCount(vertexCount);

    const std::span<const Index> inIndices{indices.cbegin(), indices.cend()};
    const std::span<const VertexAttributes> inVertices{vertices.cbegin(), vertices.cend()};
    auto * outIndex = outIndices.begin();
    auto * outVertex = outVertices.begin();
    uint32_t vertexOffset = 0;
    const auto traverseNodes = [this, &inIndices, &inVertices, &outIndex, &outVertex, &vertexOffset, vertexCount = outVertices.getCount()](const auto & self, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            const auto addVertexOffset = [&mesh, vertexOffset, vertexCount](Index i) -> Index
            {
                ASSERT(i < mesh.vertexCount);
                ASSERT(vertexOffset + i < vertexCount);
                return vertexOffset + i;
            };
            outIndex = std::ranges::transform(inIndices.subspan(mesh.indexOffset, mesh.indexCount), outIndex, addVertexOffset).out;
            outVertex = std::ranges::transform(inVertices.subspan(mesh.vertexOffset, mesh.vertexCount), outVertex, &VertexAttributes::position).out;
            vertexOffset += mesh.vertexCount;
        }
        for (size_t childIndex : node.children) {
            self(self, childIndex);
        }
    };
    traverseNodes(traverseNodes, rootNodeIndex);
    ASSERT(outIndex = outIndices.end());
    ASSERT(outVertex = outVertices.end());
}

}  // namespace scene_data
