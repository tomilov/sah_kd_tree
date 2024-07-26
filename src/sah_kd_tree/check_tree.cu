#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/check_tree.inl>
#endif

template bool sah_kd_tree::Builder<>::checkTree(const Projection<> & x, const Projection<> & y, const Projection<> & z) const;
