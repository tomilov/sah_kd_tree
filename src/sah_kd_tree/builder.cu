#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/builder.inl>
#endif

template auto sah_kd_tree::Builder<>::operator()(const Params<> & sah, Projection<> & x, Projection<> & y, Projection<> & z) -> Tree<>;
