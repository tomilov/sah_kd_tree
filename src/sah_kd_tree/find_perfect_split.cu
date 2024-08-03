#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/find_perfect_split.inl>
#endif

template void sah_kd_tree::Projection<>::findPerfectSplit(const Params<> & sah, U layerSize, const thrust::device_vector<U, Allocator<U>> & layerNodeOffset, const thrust::device_vector<U, Allocator<U>> & nodePolygonCount, const Projection & y,
                                                          const Projection & z);
