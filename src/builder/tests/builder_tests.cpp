#include "builder/builder.hpp"

#include <gtest/gtest.h>

using namespace builder;

TEST(BuilderTest, SimpleGeometry)
{
    ASSERT_TRUE(build(QStringLiteral("pointlike_triangle.obj")));
    ASSERT_TRUE(build(QStringLiteral("narrow_triangle.obj")));
    ASSERT_TRUE(build(QStringLiteral("singularity.obj")));
    ASSERT_TRUE(build(QStringLiteral("triangle.obj")));
    ASSERT_TRUE(build(QStringLiteral("aa_triangle.obj")));
    ASSERT_TRUE(build(QStringLiteral("coincident_triangles.obj")));
    ASSERT_TRUE(build(QStringLiteral("aa_parallel_non_coincident_triangles.obj")));
    ASSERT_TRUE(build(QStringLiteral("box.obj")));
    ASSERT_TRUE(build(QStringLiteral("aa_box.obj")));
    ASSERT_TRUE(build(QStringLiteral("tetrahedron.obj")));
    ASSERT_TRUE(build(QStringLiteral("box_inside_box.obj")));
}

TEST(BuilderTest, Fuzzed)
{
    ASSERT_TRUE(build(QStringLiteral("test0.obj"), false, {}, 0.285076, 0.0657117, 0.914504));
}
