#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/split_polygon.inl>
#endif

namespace sah_kd_tree
{
template void Builder<>::splitPolygon<0>(Projection<> & x, const Projection<> & y, const Projection<> & z) const;
template void Builder<>::splitPolygon<1>(Projection<> & y, const Projection<> & z, const Projection<> & x) const;
template void Builder<>::splitPolygon<2>(Projection<> & z, const Projection<> & x, const Projection<> & y) const;
}  // namespace sah_kd_tree
