#include <builder/build_from_triangles.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>

bool builder::buildSceneFromTriangles(const scene_data::Triangle * triangleBegin, const scene_data::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
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
        params.maxDepth = static_cast<sah_kd_tree::U>(maxDepth);
    }

    sah_kd_tree::Triangle triangle;
    triangle.setTriangle(triangleBegin, triangleEnd);

    sah_kd_tree::Projection x, y, z;
    sah_kd_tree::Builder builder;

    sah_kd_tree::linkTriangles(triangle, x, y, z, builder);

    sah_kd_tree::Tree tree = builder(params, x, y, z);
    return true;
}
