#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::updateSplittedPolygonNode()
{
    polygon.node.resize(polygon.count + polygon.splittedCount * 2);
    auto splittedPolygonNodeBegin = cuda::std::next(polygon.node.begin(), polygon.count);
    auto splittedPolygonNodeEnd = cuda::std::next(splittedPolygonNodeBegin, polygon.splittedCount);
    if (thrust::gather(exec, splittedPolygonNodeBegin, splittedPolygonNodeEnd, node.rightChild.cbegin(), splittedPolygonNodeEnd) != polygon.node.end()) {
        assert(false);
    }
    if (thrust::copy(exec, splittedPolygonNodeEnd, polygon.node.end(), splittedPolygonNodeBegin) != splittedPolygonNodeEnd) {
        assert(false);
    }
    polygon.node.erase(splittedPolygonNodeEnd, polygon.node.end());
}
