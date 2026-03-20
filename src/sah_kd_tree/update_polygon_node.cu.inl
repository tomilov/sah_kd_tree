#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/transform.h>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::updatePolygonNode()
{
    auto polygonBegin = thrust::make_counting_iterator<U>(0);
    auto polygonEnd = thrust::make_counting_iterator<U>(polygon.count);

    auto polygonSides = thrust::raw_pointer_cast(polygon.side.data());
    auto polygonNodes = thrust::raw_pointer_cast(polygon.node.data());

    auto nodeLeftChilds = thrust::raw_pointer_cast(node.leftChild.data());
    auto nodeRightChilds = thrust::raw_pointer_cast(node.rightChild.data());

    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());

    U layerBase = layer.base;

    const auto toPolygonNode = [polygonSides, polygonNodes, nodeLeftChilds, nodeRightChilds] __host__ __device__(U polygonIn) -> U
    {
        I polygonSide = polygonSides[polygonIn];
        U polygonNode = polygonNodes[polygonIn];
        return ((0 < polygonSide) ? nodeRightChilds : nodeLeftChilds)[polygonNode];  // splitted polygon assigned to left node
    };
    const auto isCurrentLayer = [polygonNodes, layerBase, nodeSplitDimensions] __host__ __device__(U polygonIn) -> bool
    {
        U polygonNode = polygonNodes[polygonIn];
        if (polygonNode < layerBase) {
            return false;
        }
        if (nodeSplitDimensions[polygonNode] < 0) {
            return false;
        }
        return true;
    };
    thrust::transform_if(exec, polygonBegin, polygonEnd, polygon.node.begin(), toPolygonNode, isCurrentLayer);
}
