#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/separate_splitted_polygon.inl>
#endif

template void sah_kd_tree::Builder<>::separateSplittedPolygon();
