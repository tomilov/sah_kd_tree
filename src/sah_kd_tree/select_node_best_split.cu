#include <sah_kd_tree/select_node_best_split.cu.inl>

template void sah_kd_tree::Builder<>::selectNodeBestSplit(
    const Params<> & sah,
    const Projection<> & x,
    const Projection<> & y,
    const Projection<> & z);
