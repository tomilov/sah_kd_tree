#include "sah_kd_tree.cuh"

#include <thrust/advance.h>
#include <thrust/extrema.h>

namespace SahKdTree
{
template<I dimension>
void Projection<dimension>::caluculateRootNodeBbox()
{
    auto rootBboxMinBegin = thrust::min_element(polygon.min.cbegin(), polygon.min.cend());
    node.min.assign(rootBboxMinBegin, thrust::next(rootBboxMinBegin));

    auto rootBboxMaxBegin = thrust::max_element(polygon.max.cbegin(), polygon.max.cend());
    node.max.assign(rootBboxMaxBegin, thrust::next(rootBboxMaxBegin));
}

template void Projection<0>::caluculateRootNodeBbox();
template void Projection<1>::caluculateRootNodeBbox();
template void Projection<2>::caluculateRootNodeBbox();
}  // namespace SahKdTree
