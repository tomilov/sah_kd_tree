#include "sah_kd_tree/sah_kd_tree.hpp"
#include "sah_kd_tree/utility.cuh"

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <cassert>

void sah_kd_tree::Builder::updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount)
{
    polygon.node.resize(polygonCount + splittedPolygonCount * 2);
    auto splittedPolygonNodeBegin = thrust::next(polygon.node.begin(), polygonCount);
    auto splittedPolygonNodeEnd = thrust::next(splittedPolygonNodeBegin, splittedPolygonCount);
    if (thrust::gather(splittedPolygonNodeBegin, splittedPolygonNodeEnd, node.nodeRight.cbegin(), splittedPolygonNodeEnd) != polygon.node.end()) {
        assert(false);
    }
    if (thrust::copy(splittedPolygonNodeEnd, polygon.node.end(), splittedPolygonNodeBegin) != splittedPolygonNodeEnd) {
        assert(false);
    }
    polygon.node.erase(splittedPolygonNodeEnd, polygon.node.end());
}
