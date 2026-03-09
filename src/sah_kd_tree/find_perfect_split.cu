#include <sah_kd_tree/find_perfect_split.inl>

template void sah_kd_tree::Projection<>::findPerfectSplit(
    const Params<> & sah,
    U layerSize,
    const Vector<U> & layerNodeOffset,
    const Vector<U> & nodePolygonCount,
    const Projection & y,
    const Projection & z);
