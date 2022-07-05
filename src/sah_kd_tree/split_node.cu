#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/scatter.h>

void sah_kd_tree::Projection::splitNode(I dimension, U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft,
                                        const thrust::device_vector<U> & nodeRight)
{
    auto nodeSplitPosBegin = thrust::next(nodeSplitPos.cbegin(), layerBasePrev);
    auto nodeSplitPosEnd = thrust::next(nodeSplitPos.cbegin(), layerBase);
    auto nodeSplitDimensionBegin = thrust::next(nodeSplitDimension.cbegin(), layerBasePrev);
    const auto isX = [dimension] __host__ __device__(I nodeSplitDimension) -> bool { return nodeSplitDimension == dimension; };
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeLeft.cbegin(), layerBasePrev), nodeSplitDimensionBegin, node.max.begin(), isX);
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeRight.cbegin(), layerBasePrev), nodeSplitDimensionBegin, node.min.begin(), isX);
}
