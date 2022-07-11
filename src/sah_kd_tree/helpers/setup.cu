#include <sah_kd_tree/helpers/setup.cuh>

namespace sah_kd_tree::helpers
{
void linkTriangles(const Triangle & triangle, Projection & x, Projection & y, Projection & z, Builder & builder)
{
    x.triangle.count = triangle.count;
    x.triangle.a = triangle.x.a.data();
    x.triangle.b = triangle.x.b.data();
    x.triangle.c = triangle.x.c.data();

    y.triangle.count = triangle.count;
    y.triangle.a = triangle.y.a.data();
    y.triangle.b = triangle.y.b.data();
    y.triangle.c = triangle.y.c.data();

    z.triangle.count = triangle.count;
    z.triangle.a = triangle.z.a.data();
    z.triangle.b = triangle.z.b.data();
    z.triangle.c = triangle.z.c.data();

    builder.polygon.count = triangle.count;
}
}  // namespace sah_kd_tree::helpers
