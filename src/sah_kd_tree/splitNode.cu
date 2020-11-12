#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/scatter.h>

void SahKdTree::Projection::splitNode(I dimension, U baseNodePrev, U baseNode, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft,
                                      const thrust::device_vector<U> & nodeRight)
{
    auto nodeSplitPosBegin = thrust::next(nodeSplitPos.cbegin(), baseNodePrev);
    auto nodeSplitPosEnd = thrust::next(nodeSplitPos.cbegin(), baseNode);
    auto nodeSplitDimensionBegin = thrust::next(nodeSplitDimension.cbegin(), baseNodePrev);
    auto isX = [dimension] __host__ __device__(I nodeSplitDimension) -> bool { return nodeSplitDimension == dimension; };
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeLeft.cbegin(), baseNodePrev), nodeSplitDimensionBegin, node.max.begin(), isX);
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeRight.cbegin(), baseNodePrev), nodeSplitDimensionBegin, node.min.begin(), isX);
}
