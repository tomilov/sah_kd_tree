#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/select_node_best_split.inl>
#endif

template void sah_kd_tree::Builder<>::selectNodeBestSplit(const Params<> & sah, const Projection<> & x, const Projection<> & y, const Projection<> & z);
