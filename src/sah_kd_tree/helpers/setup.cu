#include <sah_kd_tree/helpers/setup.cuh>

namespace sah_kd_tree::helpers
{
void linkTriangles(Builder & builder, const Triangles & triangles)
{
    builder.triangleCount = triangles.triangleCount;

    builder.x.triangle.a = triangles.x.a.data();
    builder.x.triangle.b = triangles.x.b.data();
    builder.x.triangle.c = triangles.x.c.data();
    builder.y.triangle.a = triangles.y.a.data();
    builder.y.triangle.b = triangles.y.b.data();
    builder.y.triangle.c = triangles.y.c.data();
    builder.z.triangle.a = triangles.z.a.data();
    builder.z.triangle.b = triangles.z.b.data();
    builder.z.triangle.c = triangles.z.c.data();
}
}  // namespace sah_kd_tree::helpers
