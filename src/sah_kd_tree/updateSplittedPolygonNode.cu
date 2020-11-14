#include "utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/gather.h>

void SahKdTree::Builder::updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount)
{
    auto splittedPolygonNodeBegin = thrust::next(polygon.node.begin(), polygonCount);
    thrust::gather(splittedPolygonNodeBegin, thrust::next(splittedPolygonNodeBegin, splittedPolygonCount), node.polygonCountRight.cbegin(), splittedPolygonNodeBegin);
}
