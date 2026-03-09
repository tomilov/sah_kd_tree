#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

#include <cassert>

namespace sah_kd_tree
{
template<typename Traits>
bool Builder<Traits>::checkBoxes(
    const Projection<Traits> & x,
    const Projection<Traits> & y,
    const Projection<Traits> & z) const
{
    U triangleCount = x.triangle.count;
    auto polygonTriangles = thrust::raw_pointer_cast(polygon.triangle.data());

    auto nodeXMins = thrust::raw_pointer_cast(x.node.min.data());
    auto nodeXMaxs = thrust::raw_pointer_cast(x.node.max.data());
    auto nodeYMins = thrust::raw_pointer_cast(y.node.min.data());
    auto nodeYMaxs = thrust::raw_pointer_cast(y.node.max.data());
    auto nodeZMins = thrust::raw_pointer_cast(z.node.min.data());
    auto nodeZMaxs = thrust::raw_pointer_cast(z.node.max.data());

    auto polygonNodes = thrust::raw_pointer_cast(polygon.node.data());

    auto polygonXMins = thrust::raw_pointer_cast(x.polygon.min.data());
    auto polygonXMaxs = thrust::raw_pointer_cast(x.polygon.max.data());
    auto polygonYMins = thrust::raw_pointer_cast(y.polygon.min.data());
    auto polygonYMaxs = thrust::raw_pointer_cast(y.polygon.max.data());
    auto polygonZMins = thrust::raw_pointer_cast(z.polygon.min.data());
    auto polygonZMaxs = thrust::raw_pointer_cast(z.polygon.max.data());

    const auto checkPolygonProjections
        = [triangleCount, polygonTriangles, polygonNodes, nodeZMaxs, polygonXMins, polygonXMaxs, polygonYMins, polygonYMaxs, polygonZMins, polygonZMaxs, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins] __host__ __device__(U polygonIn) -> bool
    {
        if (polygonTriangles[polygonIn] >= triangleCount) {
            return false;
        }

        F polygonXMin = polygonXMins[polygonIn];
        F polygonXMax = polygonXMaxs[polygonIn];
        assert(!(polygonXMax < polygonXMin));
        F polygonYMin = polygonYMins[polygonIn];
        F polygonYMax = polygonYMaxs[polygonIn];
        assert(!(polygonYMax < polygonYMin));
        F polygonZMin = polygonZMins[polygonIn];
        F polygonZMax = polygonZMaxs[polygonIn];
        assert(!(polygonZMax < polygonZMin));

        U polygonNode = polygonNodes[polygonIn];

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
    if (!thrust::all_of(exec, thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(polygon.count), checkPolygonProjections)) {
        return false;
    }
    return true;
}

template<
    typename F,
    typename U>
__host__ __device__ bool checkNodeProjection(
    const F * nodeXMins,
    const F * nodeXMaxs,
    const F * nodeYMins,
    const F * nodeYMaxs,
    const F * nodeZMins,
    const F * nodeZMaxs,
    F splitPos,
    U node,
    U leftChild,
    U rightChild)
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
    cuda::std::tuple<F, F, F, F> yz{nodeYMins[node], nodeYMaxs[node], nodeZMins[node], nodeZMaxs[node]};
    if (yz != thrust::tie(nodeYMins[leftChild], nodeYMaxs[leftChild], nodeZMins[leftChild], nodeZMaxs[leftChild])) {
        return false;
    }
    if (yz != thrust::tie(nodeYMins[rightChild], nodeYMaxs[rightChild], nodeZMins[rightChild], nodeZMaxs[rightChild])) {
        return false;
    }
    return true;
}

template<typename Traits>
bool Builder<Traits>::checkNodes(
    const Projection<Traits> & x,
    const Projection<Traits> & y,
    const Projection<Traits> & z) const
{
    auto parents = thrust::raw_pointer_cast(node.parent.data());
    auto leftChildren = thrust::raw_pointer_cast(node.leftChild.data());
    auto rightChildren = thrust::raw_pointer_cast(node.rightChild.data());
    auto splitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto splitPositions = thrust::raw_pointer_cast(node.splitPos.data());

    auto nodeXMins = thrust::raw_pointer_cast(x.node.min.data());
    auto nodeXMaxs = thrust::raw_pointer_cast(x.node.max.data());
    auto nodeYMins = thrust::raw_pointer_cast(y.node.min.data());
    auto nodeYMaxs = thrust::raw_pointer_cast(y.node.max.data());
    auto nodeZMins = thrust::raw_pointer_cast(z.node.min.data());
    auto nodeZMaxs = thrust::raw_pointer_cast(z.node.max.data());

    U polygonCount = polygon.count;
    const auto checkNode = [parents, leftChildren, rightChildren, splitDimensions, splitPositions, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs, polygonCount] __host__ __device__(U nodeIn) -> bool
    {
        I splitDimension = splitDimensions[nodeIn];
        U leftChild = leftChildren[nodeIn];
        U rightChild = rightChildren[nodeIn];
        if (splitDimension < 0) {
            if (rightChild > 0) {
                if (leftChild >= polygonCount) {
                    return false;
                }
                if (rightChild > polygonCount) {
                    return false;
                }
                if (rightChild > polygonCount - leftChild) {
                    return false;
                }
            }
            return true;
        }
        if (parents[leftChild] != nodeIn) {
            return false;
        }
        if (parents[rightChild] != nodeIn) {
            return false;
        }
        F splitPos = splitPositions[nodeIn];
        if (splitDimension == 0) {
            if (!checkNodeProjection<F, U>(nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs, splitPos, nodeIn, leftChild, rightChild)) {
                return false;
            }
        } else if (splitDimension == 1) {
            if (!checkNodeProjection<F, U>(nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs, nodeXMins, nodeXMaxs, splitPos, nodeIn, leftChild, rightChild)) {
                return false;
            }
        } else if (splitDimension == 2) {
            if (!checkNodeProjection<F, U>(nodeZMins, nodeZMaxs, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, splitPos, nodeIn, leftChild, rightChild)) {
                return false;
            }
        } else {
            assert(false);
        }
        return true;
    };
    if (!thrust::all_of(exec, thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), checkNode)) {
        return false;
    }
    return true;
}
}  // namespace sah_kd_tree
