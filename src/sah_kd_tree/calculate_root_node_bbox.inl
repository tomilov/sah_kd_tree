#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>

template<typename Traits>
void sah_kd_tree::Projection<Traits>::calculateRootNodeBbox()
{
    auto rootBboxMinBegin = thrust::min_element(polygon.min.cbegin(), polygon.min.cend());
    node.min.assign(rootBboxMinBegin, cuda::std::next(rootBboxMinBegin));

    auto rootBboxMaxBegin = thrust::max_element(polygon.max.cbegin(), polygon.max.cend());
    node.max.assign(rootBboxMaxBegin, cuda::std::next(rootBboxMaxBegin));
}
