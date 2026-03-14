#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/memory.h>

template<typename Traits>
void sah_kd_tree::linkTriangles(
    const Triangle<Traits> & triangle,
    Projection<Traits> & x,
    Projection<Traits> & y,
    Projection<Traits> & z,
    Builder<Traits> & builder)
{
    x.triangle.count = triangle.count;
    x.triangle.a = triangle.x.a;
    x.triangle.b = triangle.x.b;
    x.triangle.c = triangle.x.c;

    y.triangle.count = triangle.count;
    y.triangle.a = triangle.y.a;
    y.triangle.b = triangle.y.b;
    y.triangle.c = triangle.y.c;

    z.triangle.count = triangle.count;
    z.triangle.a = triangle.z.a;
    z.triangle.b = triangle.z.b;
    z.triangle.c = triangle.z.c;

    builder.polygon.count = triangle.count;
}
