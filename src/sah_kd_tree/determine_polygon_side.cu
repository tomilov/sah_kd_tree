#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/determine_polygon_side.inl>
#endif

namespace sah_kd_tree
{
template void Builder<>::determinePolygonSide<0>(const Projection<> & x);
template void Builder<>::determinePolygonSide<1>(const Projection<> & y);
template void Builder<>::determinePolygonSide<2>(const Projection<> & z);
}  // namespace sah_kd_tree
