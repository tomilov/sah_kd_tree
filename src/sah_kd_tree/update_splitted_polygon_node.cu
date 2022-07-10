#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <cassert>

void sah_kd_tree::Builder::updateSplittedPolygonNode(U splittedPolygonCount)
{
    polygon.node.resize(polygon.count + splittedPolygonCount * 2);
    auto splittedPolygonNodeBegin = thrust::next(polygon.node.begin(), polygon.count);
    auto splittedPolygonNodeEnd = thrust::next(splittedPolygonNodeBegin, splittedPolygonCount);
    if (thrust::gather(splittedPolygonNodeBegin, splittedPolygonNodeEnd, node.rightChild.cbegin(), splittedPolygonNodeEnd) != polygon.node.end()) {
        assert(false);
    }
    if (thrust::copy(splittedPolygonNodeEnd, polygon.node.end(), splittedPolygonNodeBegin) != splittedPolygonNodeEnd) {
        assert(false);
    }
    polygon.node.erase(splittedPolygonNodeEnd, polygon.node.end());
}
