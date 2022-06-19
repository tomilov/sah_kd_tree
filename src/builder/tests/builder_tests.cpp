#include "builder/builder.hpp"

#include <gtest/gtest.h>

using namespace builder;

TEST(BuilderTest, SimpleGeometry)
{
    EXPECT_TRUE(build("pointlike_triangle.obj"));
    EXPECT_TRUE(build("narrow_triangle.obj"));
    EXPECT_TRUE(build("singularity.obj"));
    EXPECT_TRUE(build("triangle.obj"));
    EXPECT_TRUE(build("aa_triangle.obj"));
    EXPECT_TRUE(build("coincident_triangles.obj"));
    EXPECT_TRUE(build("aa_parallel_non_coincident_triangles.obj"));
    EXPECT_TRUE(build("box.obj"));
    EXPECT_TRUE(build("aa_box.obj"));
    EXPECT_TRUE(build("tetrahedron.obj"));
    EXPECT_TRUE(build("box_inside_box.obj"));
}

TEST(BuilderTest, Fuzzed)
{
    EXPECT_TRUE(build("test0.obj", false, {}, 0.285076, 0.0657117, 0.914504));
}
