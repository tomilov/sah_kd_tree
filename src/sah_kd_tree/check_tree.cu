#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/tuple.h>

#include <cassert>

namespace sah_kd_tree
{
namespace
{
__host__ __device__ bool checkNodeProjection(const F * nodeXMins, const F * nodeXMaxs, const F * nodeYMins, const F * nodeYMaxs, const F * nodeZMins, const F * nodeZMaxs, F splitPos, U node, U leftChild, U rightChild)
{
    if (nodeXMins[leftChild] != nodeXMins[node]) {
        return false;
    }
    if (nodeXMaxs[leftChild] != splitPos) {
        return false;
    }
    if (nodeXMins[rightChild] != splitPos) {
        return false;
    }
    if (nodeXMaxs[rightChild] != nodeXMaxs[node]) {
        return false;
    }
    thrust::tuple<F, F, F, F> yz{nodeYMins[node], nodeYMaxs[node], nodeZMins[node], nodeZMaxs[node]};
    if (yz != thrust::tie(nodeYMins[leftChild], nodeYMaxs[leftChild], nodeZMins[leftChild], nodeZMaxs[leftChild])) {
        return false;
    }
    if (yz != thrust::tie(nodeYMins[rightChild], nodeYMaxs[rightChild], nodeZMins[rightChild], nodeZMaxs[rightChild])) {
        return false;
    }
    return true;
}
}  // namespace

bool Builder::checkTree(U triangleCount, U polygonCount, U nodeCount) const
{
    auto nodeXMins = x.node.min.data().get();
    auto nodeXMaxs = x.node.max.data().get();
    auto nodeYMins = y.node.min.data().get();
    auto nodeYMaxs = y.node.max.data().get();
    auto nodeZMins = z.node.min.data().get();
    auto nodeZMaxs = z.node.max.data().get();

    auto polygonNodes = polygon.node.data().get();

    auto polygonXMins = x.polygon.min.data().get();
    auto polygonXMaxs = x.polygon.max.data().get();
    auto polygonYMins = y.polygon.min.data().get();
    auto polygonYMaxs = y.polygon.max.data().get();
    auto polygonZMins = z.polygon.min.data().get();
    auto polygonZMaxs = z.polygon.max.data().get();

    const auto checkPolygon = [triangleCount, polygonNodes, nodeZMaxs, polygonXMins, polygonXMaxs, polygonYMins, polygonYMaxs, polygonZMins, polygonZMaxs, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins] __host__ __device__(U polygon) -> bool {
        F polygonXMin = polygonXMins[polygon];
        F polygonXMax = polygonXMaxs[polygon];
        assert(!(polygonXMax < polygonXMin));
        F polygonYMin = polygonYMins[polygon];
        F polygonYMax = polygonYMaxs[polygon];
        assert(!(polygonYMax < polygonYMin));
        F polygonZMin = polygonZMins[polygon];
        F polygonZMax = polygonZMaxs[polygon];
        assert(!(polygonZMax < polygonZMin));

        U polygonNode = polygonNodes[polygon];

        F nodeXMin = nodeXMins[polygonNode];
        F nodeXMax = nodeXMaxs[polygonNode];
        assert(!(nodeXMax < nodeXMin));
        F nodeYMin = nodeYMins[polygonNode];
        F nodeYMax = nodeYMaxs[polygonNode];
        assert(!(nodeYMax < nodeYMin));
        F nodeZMin = nodeZMins[polygonNode];
        F nodeZMax = nodeZMaxs[polygonNode];
        assert(!(nodeZMax < nodeZMin));

        if ((polygonXMax < nodeXMin) || (nodeXMax < polygonXMin)) {
            return false;
        }
        if ((polygonYMax < nodeYMin) || (nodeYMax < polygonYMin)) {
            return false;
        }
        if ((polygonZMax < nodeZMin) || (nodeZMax < polygonZMin)) {
            return false;
        }

        return true;
    };
    if (!thrust::all_of(thrust::make_counting_iterator<U>(triangleCount), thrust::make_counting_iterator<U>(polygonCount), checkPolygon)) {
        return false;
    }

    auto parentNodes = node.parentNode.data().get();
    auto leftChildren = node.leftChild.data().get();
    auto rightChildren = node.rightChild.data().get();
    auto splitDimensions = node.splitDimension.data().get();
    auto splitPositions = node.splitPos.data().get();

    const auto checkNode = [parentNodes, leftChildren, rightChildren, splitDimensions, splitPositions, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs] __host__ __device__(U node) -> bool {
        I splitDimension = splitDimensions[node];
        if (splitDimension < 0) {
            return true;
        }
        U leftChild = leftChildren[node];
        U rightChild = rightChildren[node];
        if (parentNodes[leftChild] != node) {
            return false;
        }
        if (parentNodes[rightChild] != node) {
            return false;
        }
        F splitPos = splitPositions[node];
        if (splitDimension == 0) {
            if (!checkNodeProjection(nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs, splitPos, node, leftChild, rightChild)) {
                return false;
            }
        } else if (splitDimension == 1) {
            if (!checkNodeProjection(nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs, nodeXMins, nodeXMaxs, splitPos, node, leftChild, rightChild)) {
                return false;
            }
        } else if (splitDimension == 2) {
            if (!checkNodeProjection(nodeZMins, nodeZMaxs, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, splitPos, node, leftChild, rightChild)) {
                return false;
            }
        } else {
            assert(false);
        }
        return true;
    };
    if (!thrust::all_of(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(nodeCount), checkNode)) {
        return false;
    }
    return true;
}
}  // namespace sah_kd_tree
