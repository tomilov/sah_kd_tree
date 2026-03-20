#include <sah_kd_tree/determine_polygon_side.cu.inl>

namespace sah_kd_tree
{
template void Builder<>::determinePolygonSide<0>(const Projection<> & x);
template void Builder<>::determinePolygonSide<1>(const Projection<> & y);
template void Builder<>::determinePolygonSide<2>(const Projection<> & z);
}  // namespace sah_kd_tree
