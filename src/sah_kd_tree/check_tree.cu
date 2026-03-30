#include <sah_kd_tree/check_tree.inl.cu>

template bool sah_kd_tree::Builder<>::checkBoxes(
    const Projection<> & x,
    const Projection<> & y,
    const Projection<> & z) const;
template bool sah_kd_tree::Builder<>::checkNodes(
    const Projection<> & x,
    const Projection<> & y,
    const Projection<> & z) const;
