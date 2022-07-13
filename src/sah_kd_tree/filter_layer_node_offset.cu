#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void sah_kd_tree::Builder::filterLayerNodeOffset()
{
    layer.nodeOffset.resize(layer.size);

    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, layer.size), thrust::next(node.polygonCount.cbegin(), layer.base), layer.nodeOffset.begin(), isNodeNotEmpty);
    layer.nodeOffset.erase(layerNodeEnd, layer.nodeOffset.end());
}
