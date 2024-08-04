#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/decouple_event_both.inl>
#endif

namespace sah_kd_tree
{
template void Projection<>::decoupleEventBoth(const Vector<I> & nodeSplitDimension, const Vector<I> & polygonSide);
}  // namespace sah_kd_tree
