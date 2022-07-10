#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void sah_kd_tree::Builder::filterLayerNodeOffset()
{
    layer.nodeOffset.resize(layer.size);

    const auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, layer.size), thrust::next(node.polygonCount.cbegin(), layer.base), layer.nodeOffset.begin(), isNodeNotEmpty);
    layer.nodeOffset.erase(layerNodeEnd, layer.nodeOffset.end());
}
