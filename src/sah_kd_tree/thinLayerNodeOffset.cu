#include "utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void SahKdTree::Builder::thinLayerNodeOffset(U baseNode, U nodeCount)
{
    layerNodeOffset.resize(nodeCount);

    auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, nodeCount), thrust::next(node.polygonCount.cbegin(), baseNode), layerNodeOffset.begin(), isNodeNotEmpty);
    layerNodeOffset.erase(layerNodeEnd, layerNodeOffset.end());
}
