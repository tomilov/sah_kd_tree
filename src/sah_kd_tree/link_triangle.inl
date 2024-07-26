#include <sah_kd_tree/sah_kd_tree.cuh>

template<typename MemoryResource>
SAH_KD_TREE_INLINE void sah_kd_tree::linkTriangles(const Triangle<MemoryResource> & triangle, Projection<MemoryResource> & x, Projection<MemoryResource> & y, Projection<MemoryResource> & z, Builder<MemoryResource> & builder)
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
