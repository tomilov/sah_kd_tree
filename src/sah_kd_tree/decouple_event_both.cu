#include <sah_kd_tree/decouple_event_both.inl.cu>

namespace sah_kd_tree
{
template void Projection<>::decoupleEventBoth(
    const Vector<I> & nodeSplitDimension,
    const Vector<I> & polygonSide);
}  // namespace sah_kd_tree
