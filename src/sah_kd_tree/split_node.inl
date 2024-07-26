#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/scatter.h>

namespace sah_kd_tree
{
template<typename Traits>
template<typename Traits::I dimension>
void Builder<Traits>::splitNode(U layerBasePrev, Projection<Traits> & projection) const
{
    auto nodeSplitPosBegin = thrust::next(node.splitPos.cbegin(), layerBasePrev);
    auto nodeSplitPosEnd = thrust::next(node.splitPos.cbegin(), layer.base);
    auto nodeSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layerBasePrev);
    const auto isCurrentProjection = [] __host__ __device__(I nodeSplitDimension) -> bool { return nodeSplitDimension == dimension; };
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), nodeSplitDimensionBegin, projection.node.max.begin(), isCurrentProjection);
    thrust::scatter_if(nodeSplitPosBegin, nodeSplitPosEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), nodeSplitDimensionBegin, projection.node.min.begin(), isCurrentProjection);
}
}  // namespace sah_kd_tree
