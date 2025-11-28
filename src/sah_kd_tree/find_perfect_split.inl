#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <limits>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Projection<Traits>::findPerfectSplit(const Params<Traits> & sah, U layerSize, const Vector<U> & layerNodeOffset, const Vector<U> & nodePolygonCount, const Projection & y, const Projection & z)
{
    {
        event.polygonCountLeft.resize(event.count);
        event.polygonCountRight.resize(event.count);

        const auto leftTriangleCountBegin = thrust::make_transform_iterator(event.kind.cbegin(), [] __host__ __device__(I eventKind) -> U { return (eventKind < 0) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.cbegin(), event.node.cend(), leftTriangleCountBegin, event.polygonCountLeft.begin());

        const auto rightTriangleCountBegin = thrust::make_transform_iterator(event.kind.crbegin(), [] __host__ __device__(I eventKind) -> U { return (0 < eventKind) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.crbegin(), event.node.crend(), rightTriangleCountBegin, event.polygonCountRight.rbegin());
    }

    layer.splitCost.resize(layerSize);
    layer.splitEvent.resize(layerSize);
    layer.splitPos.resize(layerSize);

    layer.polygonCountLeft.resize(layerSize);
    layer.polygonCountRight.resize(layerSize);
    layer.splittedPolygonCount.resize(layerSize);

    auto eventNodes = thrust::raw_pointer_cast(event.node.data());
    auto eventPositions = thrust::raw_pointer_cast(event.pos.data());
    auto eventKinds = thrust::raw_pointer_cast(event.kind.data());
    auto polygonCountLefts = thrust::raw_pointer_cast(event.polygonCountLeft.data());  // lefts, rights is just notation
    auto polygonCountRights = thrust::raw_pointer_cast(event.polygonCountRight.data());
    auto nodePolygonCounts = thrust::raw_pointer_cast(nodePolygonCount.data());

    auto nodeXMins = thrust::raw_pointer_cast(node.min.data());
    auto nodeXMaxs = thrust::raw_pointer_cast(node.max.data());
    auto nodeYMins = thrust::raw_pointer_cast(y.node.min.data());
    auto nodeYMaxs = thrust::raw_pointer_cast(y.node.max.data());
    auto nodeZMins = thrust::raw_pointer_cast(z.node.min.data());
    auto nodeZMaxs = thrust::raw_pointer_cast(z.node.max.data());

    auto perfectSplitBegin = thrust::make_zip_iterator(layer.splitCost.begin(), layer.splittedPolygonCount.begin(), layer.splitPos.begin(), layer.polygonCountLeft.begin(), layer.polygonCountRight.begin(), layer.splitEvent.begin());
    auto perfectSplitOutputBegin = thrust::make_permutation_iterator(perfectSplitBegin, layerNodeOffset.cbegin());
    using PerfectSplitType = cuda::std::iter_value_t<decltype(perfectSplitOutputBegin)>;
    const auto toPerfectSplit = [sah, eventNodes, eventPositions, eventKinds, polygonCountLefts, polygonCountRights, nodePolygonCounts, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs] __host__ __device__(U event) -> PerfectSplitType
    {
        U eventNode = eventNodes[event];
        F min = nodeXMins[eventNode], max = nodeXMaxs[eventNode];
        F splitPos = eventPositions[event];
        assert(!(splitPos < min));
        assert(!(max < splitPos));
        U polygonCountLeft = polygonCountLefts[event];
        U polygonCountRight = polygonCountRights[event];
        U polygonCount = nodePolygonCounts[eventNode];
        assert(polygonCountLeft <= polygonCount);
        assert(polygonCountRight <= polygonCount);
        U splittedPolygonCount = polygonCountLeft + polygonCountRight - polygonCount;
        U splitEvent = event;
        F splitCost = std::numeric_limits<F>::infinity();
        if (!(min < max)) {
            return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
        }
        F l = splitPos - min, r = max - splitPos;
        I eventKind = eventKinds[event];
        if (eventKind < 0) {
            assert(0 != polygonCountLeft);
            ++splitEvent;
        } else if (eventKind == 0) {
            if ((l < r) ? (polygonCountLeft != 0) : (polygonCountRight == 0)) {
                ++polygonCountLeft;
                ++splitEvent;
            } else {
                ++polygonCountRight;
            }
        } else {
            assert(0 != polygonCountRight);
        }
        F emptinessFactor(1);
        if (polygonCountLeft == 0) {
            assert(polygonCountRight != 0);
            if (!(min < splitPos)) {
                return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
            }
            emptinessFactor = sah.emptinessFactor;
        } else if (polygonCountRight == polygonCount) {
            return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
        } else if (polygonCountRight == 0) {
            if (!(splitPos < max)) {
                return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
            }
            emptinessFactor = sah.emptinessFactor;
        } else if (polygonCountLeft == polygonCount) {
            return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
        }
        F x = max - min;
        F y = nodeYMaxs[eventNode] - nodeYMins[eventNode];
        assert(static_cast<F>(0) <= y);
        F z = nodeZMaxs[eventNode] - nodeZMins[eventNode];
        assert(static_cast<F>(0) <= z);
        F halfArea = y * z;
        if (static_cast<F>(0) < halfArea) {
            F halfPerimeter = y + z;
            assert(static_cast<F>(0) < halfPerimeter);
            splitCost = (static_cast<F>(polygonCountLeft) * (halfArea + halfPerimeter * l) + static_cast<F>(polygonCountRight) * (halfArea + halfPerimeter * r)) / (halfArea + halfPerimeter * x);
        } else {
            splitCost = (static_cast<F>(polygonCountLeft) * l + static_cast<F>(polygonCountRight) * r) / x;
        }
        splitCost *= sah.intersectionCost;
        splitCost += sah.traversalCost;
        splitCost *= emptinessFactor;
        return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
    };
    auto perfectSplitValueBegin = thrust::make_transform_iterator(thrust::make_counting_iterator<U>(0), toPerfectSplit);
    [[maybe_unused]] auto ends = thrust::reduce_by_key(event.node.cbegin(), event.node.cend(), perfectSplitValueBegin, thrust::make_discard_iterator(), perfectSplitOutputBegin, cuda::std::equal_to<U>{}, cuda::minimum<PerfectSplitType>{});
    assert(ends.first == thrust::make_discard_iterator(layerNodeOffset.size()));
}
