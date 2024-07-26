#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/memory.h>

template<typename Traits>
SAH_KD_TREE_INLINE void sah_kd_tree::Builder<Traits>::updatePolygonNode()
{
    auto polygonBegin = thrust::make_counting_iterator<U>(0);
    auto polygonEnd = thrust::make_counting_iterator<U>(polygon.count);

    auto polygonSides = thrust::raw_pointer_cast(polygon.side.data());
    auto polygonNodes = thrust::raw_pointer_cast(polygon.node.data());

    auto nodeLeftChilds = thrust::raw_pointer_cast(node.leftChild.data());
    auto nodeRightChilds = thrust::raw_pointer_cast(node.rightChild.data());

    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());

    U layerBase = layer.base;

    const auto toPolygonNode = [polygonSides, polygonNodes, nodeLeftChilds, nodeRightChilds] __host__ __device__(U polygon) -> U {
        I polygonSide = polygonSides[polygon];
        U polygonNode = polygonNodes[polygon];
        return ((0 < polygonSide) ? nodeRightChilds : nodeLeftChilds)[polygonNode];  // splitted polygon assigned to left node
    };
    const auto isCurrentLayer = [polygonNodes, layerBase, nodeSplitDimensions] __host__ __device__(U polygon) -> bool {
        U polygonNode = polygonNodes[polygon];
        if (polygonNode < layerBase) {
            return false;
        }
        if (nodeSplitDimensions[polygonNode] < 0) {
            return false;
        }
        return true;
    };
    thrust::transform_if(polygonBegin, polygonEnd, polygon.node.begin(), toPolygonNode, isCurrentLayer);
}
