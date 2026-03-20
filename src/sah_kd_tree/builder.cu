#include <sah_kd_tree/builder.cu.inl>

template bool sah_kd_tree::Builder<>::build<>(
    const Progress & progress,
    const Params<> & sah,
    Projection<> & x,
    Projection<> & y,
    Projection<> & z,
    Tree<> & tree);
