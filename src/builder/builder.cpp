#include <builder/builder.hpp>

#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

#include <QDebug>

Q_LOGGING_CATEGORY(builderLog, "builder")

namespace {

bool buildSceneFromTriangles(const QVector<scene_loader::Triangle> & t, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    using sah_kd_tree::U;
    using sah_kd_tree::F;

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
    sah_kd_tree::helpers::setTriangles(triangles, std::data(t), std::data(t) + std::size(t));

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

    qCDebug(builderLog) << QStringLiteral("SAH k-D tree depth = %1").arg(tree.depth);
    if (tree.depth > U(std::size(t))) {
        return false;
    }
    return true;
}

}  // namespace

bool builder::buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    if (!sceneLoader.load(sceneFileName)) {
        return false;
    }
    return buildSceneFromTriangles(sceneLoader.triangle, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool builder::buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    if (!sceneLoader.cachingLoad(sceneFileName, cachePath)) {
        return false;
    }
    return buildSceneFromTriangles(sceneLoader.triangle, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
