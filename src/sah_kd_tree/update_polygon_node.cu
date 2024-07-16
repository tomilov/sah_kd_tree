#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

void sah_kd_tree::Builder::updatePolygonNode()
{
    auto polygonBegin = thrust::make_counting_iterator<U>(0);
    auto polygonEnd = thrust::make_counting_iterator<U>(polygon.count);

    auto polygonSides = polygon.side.data().get();
    auto polygonNodes = polygon.node.data().get();

    auto nodeLeftChilds = node.leftChild.data().get();
    auto nodeRightChilds = node.rightChild.data().get();

    auto nodeSplitDimensions = node.splitDimension.data().get();

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
