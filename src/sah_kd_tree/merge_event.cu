#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/merge_event.inl>
#endif

template void sah_kd_tree::Projection<>::mergeEvent(U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & polygonNode, const thrust::device_vector<U> & splittedPolygon);
