#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/set_node_count.inl>
#endif

template void sah_kd_tree::Builder<>::setNodeCount(Projection<> & x, Projection<> & y, Projection<> & z) const;
