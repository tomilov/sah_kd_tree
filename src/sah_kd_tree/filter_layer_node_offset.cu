#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/filter_layer_node_offset.inl>
#endif

template void sah_kd_tree::Builder<>::filterLayerNodeOffset();
