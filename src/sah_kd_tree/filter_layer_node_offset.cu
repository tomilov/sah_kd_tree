#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void sah_kd_tree::Builder::filterLayerNodeOffset()
{
    layer.nodeOffset.resize(layer.size);

    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::make_counting_iterator<U>(layer.size);
    auto layerNodeOffsetEnd = thrust::copy_if(layerNodeBegin, layerNodeEnd, thrust::next(node.polygonCount.cbegin(), layer.base), layer.nodeOffset.begin(), isNodeNotEmpty);
    layer.nodeOffset.erase(layerNodeOffsetEnd, layer.nodeOffset.end());
}
