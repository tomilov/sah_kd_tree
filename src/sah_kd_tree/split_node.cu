#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/split_node.inl>
#endif

namespace sah_kd_tree
{
template void Builder<>::splitNode<0>(U layerBasePrev, Projection<> & x) const;
template void Builder<>::splitNode<1>(U layerBasePrev, Projection<> & y) const;
template void Builder<>::splitNode<2>(U layerBasePrev, Projection<> & z) const;
}  // namespace sah_kd_tree
