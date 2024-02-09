#include <scene/scene.hpp>
#include <utils/assert.hpp>

namespace scene
{

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

void Scene::updateAABBs()
{
    for (scene::Node & node : nodes) {
        for (size_t meshIndex : node.meshes) {
            const auto & mesh = meshes.at(meshIndex);
            node.aabb.min = glm::min(node.aabb.min, mesh.aabb.min);
            node.aabb.max = glm::max(node.aabb.max, mesh.aabb.max);
        }
        aabb.min = glm::min(aabb.min, node.aabb.min);
        aabb.max = glm::max(aabb.max, node.aabb.max);
    }
}

utils::MemArray<Triangle> Scene::makeTriangles() const
{
    size_t vertexCount = 0;
    for (const Mesh & mesh : meshes) {
        INVARIANT((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
        vertexCount += mesh.indexCount;
    }

    utils::MemArray<Triangle> triangles{vertexCount / 3};
    auto t = triangles.begin();
    auto v = vertices.begin();
    for (const Mesh & mesh : meshes) {
        auto index = indices.begin();
        std::advance(index, mesh.indexOffset);
        auto endIndex = std::next(index, mesh.indexCount);
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

utils::MemArray<Triangle> Scene::makeTriangles(size_t rootNodeIndex) const
{
    size_t vertexCount = 0;
    const auto countTriangles = [this, &vertexCount](const auto & countTriangles, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            INVARIANT((mesh.indexCount % 3) == 0, "{}", mesh.indexCount);
            vertexCount += mesh.indexCount;
        }
        for (size_t childIndex : node.children) {
            countTriangles(countTriangles, childIndex);
        }
    };
    countTriangles(countTriangles, rootNodeIndex);

    utils::MemArray<Triangle> triangles{vertexCount / 3};
    auto t = triangles.begin();
    auto v = vertices.begin();
    const auto traverseNodes = [this, &t, &triangles, v](const auto & traverseNodes, size_t nodeIndex) -> void
    {
        const Node & node = nodes[nodeIndex];
        for (size_t m : node.meshes) {
            const Mesh & mesh = meshes[m];
            auto index = indices.begin();
            std::advance(index, mesh.indexOffset);
            auto endIndex = std::next(index, mesh.indexCount);
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
            traverseNodes(traverseNodes, childIndex);
        }
    };
    traverseNodes(traverseNodes, rootNodeIndex);
    ASSERT(t == triangles.end());
    return triangles;
}

}  // namespace scene
