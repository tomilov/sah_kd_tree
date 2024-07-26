#if SAH_KD_TREE_HEADER_ONLY
#error "!"
#else
#include <sah_kd_tree/generate_initial_event.inl>
#endif

template void sah_kd_tree::Projection<>::generateInitialEvent();
