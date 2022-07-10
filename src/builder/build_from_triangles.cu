#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <builder/build_from_triangles.hpp>

bool builder::buildSceneFromTriangles(const scene_loader::Triangle * triangleBegin, const scene_loader::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    sah_kd_tree::Params params;
    if (emptinessFactor > 0.0f) {
        params.emptinessFactor = emptinessFactor;
    }
    if (traversalCost > 0.0f) {
        params.traversalCost = traversalCost;
    }
    if (intersectionCost > 0.0f) {
        params.intersectionCost = intersectionCost;
    }
    if (maxDepth > 0) {
        params.maxDepth = maxDepth;
    }

    sah_kd_tree::helpers::Triangles triangles;
    triangles.setTriangles(triangleBegin, triangleEnd);

    sah_kd_tree::Builder builder;
    sah_kd_tree::helpers::linkTriangles(builder, triangles);

    sah_kd_tree::Tree tree = builder(params);
    return true;
}
