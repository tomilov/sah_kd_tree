#include <builder/build_from_triangles.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>

#include <gtest/gtest.h>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>
#include <QtCore/QtLogging>

using namespace Qt::StringLiterals;

namespace
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(builderTest)
Q_LOGGING_CATEGORY(builderTest, "builder.test")
}  // namespace

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0)
{
    scene_data::SceneData sceneData;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!scene_loader::load(sceneData, sceneFileInfo)) {
        qCDebug(builderTest).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
        return false;
    }
    auto triangles = sceneData.makeTriangles();
    return builder::buildSceneFromTriangles(triangles.begin(), triangles.end(), emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0)
{
    scene_data::SceneData sceneData;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!scene_loader::cachingLoad(sceneData, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
        qCDebug(builderTest).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
        return false;
    }
    auto triangles = sceneData.makeTriangles();
    return builder::buildSceneFromTriangles(triangles.begin(), triangles.end(), emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

}  // namespace

TEST(Builder, SimpleGeometry)
{
    EXPECT_TRUE(buildSceneFromFile("pointlike_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("singularity.obj"));
    EXPECT_TRUE(buildSceneFromFile("narrow_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("coincident_triangles.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_parallel_non_coincident_triangles.obj"));
    EXPECT_TRUE(buildSceneFromFile("box.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_box.obj"));
    EXPECT_TRUE(buildSceneFromFile("tetrahedron.obj"));
    EXPECT_TRUE(buildSceneFromFile("box_inside_box.obj"));
}

TEST(Builder, Fuzzed)
{
    EXPECT_TRUE(buildSceneFromFile("test0.obj", 0.285076, 0.0657117, 0.914504));
    EXPECT_TRUE(buildSceneFromFile("test1.obj", 0x1.a538900000000p-2, 0x1.ddf3b40000000p-5, 0x1.ecdd120000000p-4));
    EXPECT_TRUE(buildSceneFromFile("test2.obj", 0.7149041295051575, 0.060609497129917145, 0.17161905765533447));
    EXPECT_TRUE(buildSceneFromFile("test3.obj", 0.7149041295051575, 0.18199801445007324, 0.3812173902988434));
    EXPECT_TRUE(buildSceneFromFile("triangle_of_degenerate_triangles.obj", 0x1.2222aa0000000p-1, 0x1.5464960000000p-3, 0x1.152a640000000p-1));
}

TEST(Builder, DISABLED_AllScenes)
{
    auto scenes = QDir::current().entryList(QStringList() << "*.obj", QDir::Files, QDir::Size | QDir::Reversed);
    for (const auto & fileName : scenes) {  // clazy:exclude=range-loop-detach
        EXPECT_TRUE(buildSceneFromFileOrCache(fileName, {} /* cachePath */));
    }
}
