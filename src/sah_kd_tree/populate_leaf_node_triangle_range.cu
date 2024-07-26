#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/populate_leaf_node_triangle_range.inl>
#endif

template void sah_kd_tree::Builder<>::populateLeafNodeTriangleRange();
