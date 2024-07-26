#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/calculate_rope.inl>
#endif

namespace sah_kd_tree
{
template void Builder<>::calculateRope<0, false>(Projection<> & x, const Projection<> & y, const Projection<> & z) const;
template void Builder<>::calculateRope<0, true>(Projection<> & x, const Projection<> & y, const Projection<> & z) const;
template void Builder<>::calculateRope<1, false>(Projection<> & y, const Projection<> & z, const Projection<> & x) const;
template void Builder<>::calculateRope<1, true>(Projection<> & y, const Projection<> & z, const Projection<> & x) const;
template void Builder<>::calculateRope<2, false>(Projection<> & z, const Projection<> & x, const Projection<> & y) const;
template void Builder<>::calculateRope<2, true>(Projection<> & z, const Projection<> & x, const Projection<> & y) const;
}
