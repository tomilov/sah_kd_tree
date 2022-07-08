#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/scatter.h>

namespace sah_kd_tree
{
template<I dimension>
void Projection::splitNode(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft, const thrust::device_vector<U> & nodeRight)
{
    auto nodeSplitPosBegin = thrust::next(nodeSplitPos.cbegin(), layerBasePrev);
    auto nodeSplitPosEnd = thrust::next(nodeSplitPos.cbegin(), layerBase);
    auto nodeSplitDimensionBegin = thrust::next(nodeSplitDimension.cbegin(), layerBasePrev);
    const auto isX = [] __host__ __device__(I nodeSplitDimension) -> bool { return nodeSplitDimension == dimension; };
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeLeft.cbegin(), layerBasePrev), nodeSplitDimensionBegin, node.max.begin(), isX);
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(nodeRight.cbegin(), layerBasePrev), nodeSplitDimensionBegin, node.min.begin(), isX);
}

template void Projection::splitNode<0>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft,
                                       const thrust::device_vector<U> & nodeRight);
template void Projection::splitNode<1>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft,
                                       const thrust::device_vector<U> & nodeRight);
template void Projection::splitNode<2>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft,
                                       const thrust::device_vector<U> & nodeRight);
}  // namespace sah_kd_tree
