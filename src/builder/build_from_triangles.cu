#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <builder/build_from_triangles.hpp>

bool builder::buildSceneFromTriangles(const scene_loader::Triangle * triangleBegin, const scene_loader::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    sah_kd_tree::Params params;
    if (emptinessFactor > 0.0f) {
        params.emptinessFactor = sah_kd_tree::F(emptinessFactor);
    }
    if (traversalCost > 0.0f) {
        params.traversalCost = sah_kd_tree::F(traversalCost);
    }
    if (intersectionCost > 0.0f) {
        params.intersectionCost = sah_kd_tree::F(intersectionCost);
    }
    if (maxDepth > 0) {
        params.maxDepth = sah_kd_tree::U(maxDepth);
    }

    sah_kd_tree::helpers::Triangles triangles;
    sah_kd_tree::helpers::setTriangles(triangles, triangleBegin, triangleEnd);

    sah_kd_tree::Builder builder;
    sah_kd_tree::helpers::linkTriangles(builder, triangles);

    sah_kd_tree::Tree tree = builder(params);

    // qCDebug(builderLog) << QStringLiteral("SAH k-D tree depth = %1").arg(tree.depth);
    if (builder.triangleCount < tree.depth) {
        return false;
    }
    return true;
}
