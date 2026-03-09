#include <sah_kd_tree/link_triangle.inl>

template void sah_kd_tree::linkTriangles(
    const Triangle<> & triangle,
    Projection<> & x,
    Projection<> & y,
    Projection<> & z,
    Builder<> & builder);
