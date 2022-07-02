#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>

void sah_kd_tree::Projection::calculateRootNodeBbox()
{
    auto rootBboxMinBegin = thrust::min_element(polygon.min.cbegin(), polygon.min.cend());
    node.min.assign(rootBboxMinBegin, thrust::next(rootBboxMinBegin));

    auto rootBboxMaxBegin = thrust::max_element(polygon.max.cbegin(), polygon.max.cend());
    node.max.assign(rootBboxMaxBegin, thrust::next(rootBboxMaxBegin));
}
