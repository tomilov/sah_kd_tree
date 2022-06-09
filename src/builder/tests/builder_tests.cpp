#include "builder/builder.hpp"

#include <gtest/gtest.h>

using namespace builder;

TEST(BuilderTest, SimpleGeometry)
{
    ASSERT_TRUE(build("pointlike_triangle.obj"));
    ASSERT_TRUE(build("narrow_triangle.obj"));
    ASSERT_TRUE(build("singularity.obj"));
    ASSERT_TRUE(build("triangle.obj"));
    ASSERT_TRUE(build("aa_triangle.obj"));
    ASSERT_TRUE(build("coincident_triangles.obj"));
    ASSERT_TRUE(build("aa_parallel_non_coincident_triangles.obj"));
    ASSERT_TRUE(build("box.obj"));
    ASSERT_TRUE(build("aa_box.obj"));
    ASSERT_TRUE(build("tetrahedron.obj"));
    ASSERT_TRUE(build("box_inside_box.obj"));
}

TEST(BuilderTest, Fuzzed)
{
    ASSERT_TRUE(build("test0.obj", false, {}, 0.285076, 0.0657117, 0.914504));
}
