#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <builder/build_from_triangles.hpp>

bool builder::buildSceneFromTriangles(const scene_loader::Triangle * triangleBegin, const scene_loader::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    using sah_kd_tree::F;
    using sah_kd_tree::U;

    sah_kd_tree::Params params;
    if (emptinessFactor > 0.0f) {
        params.emptinessFactor = F(emptinessFactor);
    }
    if (traversalCost > 0.0f) {
        params.traversalCost = F(traversalCost);
    }
    if (intersectionCost > 0.0f) {
        params.intersectionCost = F(intersectionCost);
    }
    if (maxDepth > 0) {
        params.maxDepth = U(maxDepth);
    }

    sah_kd_tree::helpers::Triangles triangles;
    sah_kd_tree::helpers::setTriangles(triangles, triangleBegin, triangleEnd);

    sah_kd_tree::Builder builder;
    {
        builder.triangleCount = triangles.triangleCount;

        builder.x.triangle.a = triangles.x.a.data();
        builder.x.triangle.b = triangles.x.c.data();
        builder.x.triangle.c = triangles.x.b.data();
        builder.y.triangle.a = triangles.y.a.data();
        builder.y.triangle.b = triangles.y.c.data();
        builder.y.triangle.c = triangles.y.b.data();
        builder.z.triangle.a = triangles.z.a.data();
        builder.z.triangle.b = triangles.z.c.data();
        builder.z.triangle.c = triangles.z.b.data();
    }

    sah_kd_tree::Tree tree = builder(params);

    // qCDebug(builderLog) << QStringLiteral("SAH k-D tree depth = %1").arg(tree.depth);
    if (tree.depth > builder.triangleCount) {
        return false;
    }
    return true;
}
