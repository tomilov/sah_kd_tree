#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/decouple_event_both.inl>
#endif

namespace sah_kd_tree
{
template void Projection<>::decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide);
}  // namespace sah_kd_tree
