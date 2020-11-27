#include "Utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void SahKdTree::Builder::thinLayerNodeOffset(U layerBase, U layerSize)
{
    layerNodeOffset.resize(layerSize);

    auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, layerSize), thrust::next(node.polygonCount.cbegin(), layerBase), layerNodeOffset.begin(), isNodeNotEmpty);
    layerNodeOffset.erase(layerNodeEnd, layerNodeOffset.end());
}
