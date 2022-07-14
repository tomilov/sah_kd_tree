#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/type_traits.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <limits>

#include <cassert>

void sah_kd_tree::Projection::findPerfectSplit(const Params & sah, U layerSize, const thrust::device_vector<U> & layerNodeOffset, const thrust::device_vector<U> & nodePolygonCount, const Projection & y, const Projection & z)
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

    auto eventNodes = event.node.data().get();
    auto eventPositions = event.pos.data().get();
    auto eventKinds = event.kind.data().get();
    auto polygonCountLefts = event.polygonCountLeft.data().get();  // lefts, rights is just notation
    auto polygonCountRights = event.polygonCountRight.data().get();
    auto nodePolygonCounts = nodePolygonCount.data().get();

    auto nodeXMins = node.min.data().get();
    auto nodeXMaxs = node.max.data().get();
    auto nodeYMins = y.node.min.data().get();
    auto nodeYMaxs = y.node.max.data().get();
    auto nodeZMins = z.node.min.data().get();
    auto nodeZMaxs = z.node.max.data().get();

    auto perfectSplitBegin = thrust::make_zip_iterator(layer.splitCost.begin(), layer.splittedPolygonCount.begin(), layer.splitPos.begin(), layer.polygonCountLeft.begin(), layer.polygonCountRight.begin(), layer.splitEvent.begin());
    auto perfectSplitOutputBegin = thrust::make_permutation_iterator(perfectSplitBegin, layerNodeOffset.cbegin());
    using PerfectSplitType = IteratorValueType<decltype(perfectSplitOutputBegin)>;
    const auto toPerfectSplit =
        [sah, eventNodes, eventPositions, eventKinds, polygonCountLefts, polygonCountRights, nodePolygonCounts, nodeXMins, nodeXMaxs, nodeYMins, nodeYMaxs, nodeZMins, nodeZMaxs] __host__ __device__(U event) -> PerfectSplitType {
        U eventNode = eventNodes[event];
        F min = nodeXMins[eventNode], max = nodeXMaxs[eventNode];
        F splitPos = eventPositions[event];
        assert(!(splitPos < min));
        assert(!(max < splitPos));
        if (!(min < max)) {
            return {std::numeric_limits<F>::infinity(), 0};
        }
        F l = splitPos - min, r = max - splitPos;
        U polygonCountLeft = polygonCountLefts[event];
        U polygonCountRight = polygonCountRights[event];
        U splitEvent = event;
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
        U polygonCount = nodePolygonCounts[eventNode];
        F emptinessFactor(1);
        if (polygonCountLeft == 0) {
            assert(polygonCountRight != 0);
            if (!(min < splitPos)) {
                return {std::numeric_limits<F>::infinity(), 0};
            }
            emptinessFactor = sah.emptinessFactor;
        } else if (polygonCountRight == polygonCount) {
            return {std::numeric_limits<F>::infinity(), 0};
        } else if (polygonCountRight == 0) {
            if (!(splitPos < max)) {
                return {std::numeric_limits<F>::infinity(), 0};
            }
            emptinessFactor = sah.emptinessFactor;
        } else if (polygonCountLeft == polygonCount) {
            return {std::numeric_limits<F>::infinity(), 0};
        }
        F x = max - min;
        F y = nodeYMaxs[eventNode] - nodeYMins[eventNode];
        F z = nodeZMaxs[eventNode] - nodeZMins[eventNode];
        F area = y * z;  // half area
        F splitCost;
        if (F(0) < area) {
            F perimeter = y + z;  // half perimeter
            assert(F(0) < perimeter);
            splitCost = (polygonCountLeft * (area + perimeter * l) + polygonCountRight * (area + perimeter * r)) / (area + perimeter * x);
        } else {
            splitCost = (polygonCountLeft * l + polygonCountRight * r) / x;
        }
        splitCost *= sah.intersectionCost;
        splitCost += sah.traversalCost;
        splitCost *= emptinessFactor;
        U splittedPolygonCount = polygonCountLeft + polygonCountRight - polygonCount;
        return {splitCost, splittedPolygonCount, splitPos, polygonCountLeft, polygonCountRight, splitEvent};
    };
    auto perfectSplitValueBegin = thrust::make_transform_iterator(thrust::make_counting_iterator<U>(0), toPerfectSplit);
    [[maybe_unused]] auto ends = thrust::reduce_by_key(event.node.cbegin(), event.node.cend(), perfectSplitValueBegin, thrust::make_discard_iterator(), perfectSplitOutputBegin, thrust::equal_to<U>{}, thrust::minimum<PerfectSplitType>{});
    assert(ends.first == thrust::make_discard_iterator(layerNodeOffset.size()));
}
