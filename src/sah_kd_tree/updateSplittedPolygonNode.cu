#include "utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <cassert>

void SahKdTree::Builder::updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount)
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
