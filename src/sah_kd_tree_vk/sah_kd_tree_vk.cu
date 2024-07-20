#include <sah_kd_tree/sah_kd_tree.cuh>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <sah_kd_tree_vk/sah_kd_tree_vk.hpp>

namespace sah_kd_tree_vk
{
namespace
{

}

void Tree::Settings::check() const
{
    ASSERT(emptinessFactor > 0.0f);
    ASSERT(traversalCost > 0.0f);
    ASSERT(intersectionCost > 0.0f);
}

struct Tree::Impl : utils::NonCopyable
{
    Impl(const scene_data::SceneData & sceneData, const Settings & settings)
    {
        settings.check();
        auto triangles = sceneData.makeTriangles();
        sah_kd_tree::Tree tree = makeTree(triangles.begin(), triangles.end(), settings);
    }

    [[nodiscard]] static sah_kd_tree::Tree makeTree(const scene_data::Triangle * triangleBegin, const scene_data::Triangle * triangleEnd, const Settings & settings)
    {
        sah_kd_tree::Params params = {
            .emptinessFactor = settings.emptinessFactor,
            .traversalCost = settings.traversalCost,
            .intersectionCost = settings.intersectionCost,
            .maxDepth = settings.maxDepth,
        };

        sah_kd_tree::Triangle triangle;
        triangle.setTriangle(triangleBegin, triangleEnd);

        sah_kd_tree::Projection x, y, z;
        sah_kd_tree::Builder builder;

        sah_kd_tree::linkTriangles(triangle, x, y, z, builder);

        return builder(params, x, y, z);
    }
};

Tree::Tree(const scene_data::SceneData & sceneData, const Settings & settings)
    : impl_{sceneData, settings}
{}

Tree::~Tree() = default;

}  // namespace sah_kd_tree_vk
