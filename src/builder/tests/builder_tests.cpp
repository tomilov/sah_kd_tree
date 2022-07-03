#include <builder/builder.hpp>

#include <QDir>

#include <gtest/gtest.h>

using builder::buildSceneFromFile;
using builder::buildSceneFromFileOrCache;

TEST(BuilderTest, SimpleGeometry)
{
    EXPECT_TRUE(buildSceneFromFile("pointlike_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("narrow_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("singularity.obj"));
    EXPECT_TRUE(buildSceneFromFile("triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_triangle.obj"));
    EXPECT_TRUE(buildSceneFromFile("coincident_triangles.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_parallel_non_coincident_triangles.obj"));
    EXPECT_TRUE(buildSceneFromFile("box.obj"));
    EXPECT_TRUE(buildSceneFromFile("aa_box.obj"));
    EXPECT_TRUE(buildSceneFromFile("tetrahedron.obj"));
    EXPECT_TRUE(buildSceneFromFile("box_inside_box.obj"));
}

TEST(BuilderTest, Fuzzed)
{
    EXPECT_TRUE(buildSceneFromFile("test0.obj", 0.285076, 0.0657117, 0.914504));
    EXPECT_TRUE(buildSceneFromFile("test1.obj", 0x1.a538900000000p-2, 0x1.ddf3b40000000p-5, 0x1.ecdd120000000p-4));
    EXPECT_TRUE(buildSceneFromFile("test2.obj", 0.7149041295051575, 0.060609497129917145, 0.17161905765533447));
    EXPECT_TRUE(buildSceneFromFile("test3.obj", 0.7149041295051575, 0.18199801445007324, 0.3812173902988434));
}

TEST(BuilderTest, DISABLED_AllScenes)
{
    auto scenes = QDir::current().entryList(QStringList() << "*.obj", QDir::Files, QDir::Size | QDir::Reversed);
    for (const auto & fileName : scenes) {  // clazy:exclude=range-loop-detach
        EXPECT_TRUE(buildSceneFromFileOrCache(fileName, {}));
    }
}
