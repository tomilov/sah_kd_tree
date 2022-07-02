#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <limits>

#include <cassert>

void sah_kd_tree::Projection::findPerfectSplit(const Params & sah, U layerSize, const thrust::device_vector<U> & layerNodeOffset, const thrust::device_vector<U> & nodePolygonCount, const Projection & y, const Projection & z)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    {
        event.polygonCountLeft.resize(eventCount);
        event.polygonCountRight.resize(eventCount);

        auto leftTriangleCountBegin = thrust::make_transform_iterator(event.kind.cbegin(), [] __host__ __device__(I eventKind) -> U { return (eventKind < 0) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.cbegin(), event.node.cend(), leftTriangleCountBegin, event.polygonCountLeft.begin());

        auto rightTriangleCountBegin = thrust::make_transform_iterator(event.kind.crbegin(), [] __host__ __device__(I eventKind) -> U { return (0 < eventKind) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.crbegin(), event.node.crend(), rightTriangleCountBegin, event.polygonCountRight.rbegin());
    }
    timer(" findPerfectSplit 2 * exclusive_scan_by_key");  // 4.131ms

    layer.splitCost.resize(layerSize);
    layer.splitEvent.resize(layerSize);
    layer.splitPos.resize(layerSize);

    layer.polygonCountLeft.resize(layerSize);
    layer.polygonCountRight.resize(layerSize);
    timer(" findPerfectSplit resize");  // 0.640ms

    auto nodeLimitsBegin = thrust::make_zip_iterator(thrust::make_tuple(node.min.cbegin(), node.max.cbegin(), y.node.min.cbegin(), y.node.max.cbegin(), z.node.min.cbegin(), z.node.max.cbegin()));
    auto nodeBboxBegin = thrust::make_permutation_iterator(nodeLimitsBegin, event.node.cbegin());
    using NodeBboxType = IteratorValueType<decltype(nodeBboxBegin)>;
    auto splitEventBegin = thrust::make_counting_iterator<U>(0);
    auto polygonCount = thrust::make_permutation_iterator(nodePolygonCount.cbegin(), event.node.cbegin());
    auto perfectSplitInputBegin = thrust::make_zip_iterator(thrust::make_tuple(nodeBboxBegin, event.pos.cbegin(), event.kind.cbegin(), splitEventBegin, polygonCount, event.polygonCountLeft.cbegin(), event.polygonCountRight.cbegin()));
    auto perfectSplitBegin = thrust::make_zip_iterator(thrust::make_tuple(layer.splitCost.begin(), layer.splitEvent.begin(), layer.splitPos.begin(), layer.polygonCountLeft.begin(), layer.polygonCountRight.begin()));
    auto perfectSplitOutputBegin = thrust::make_permutation_iterator(perfectSplitBegin, layerNodeOffset.cbegin());
    using PerfectSplitType = IteratorValueType<decltype(perfectSplitOutputBegin)>;
    auto toPerfectSplit = [sah] __host__ __device__(NodeBboxType nodeBbox, F splitPos, I eventKind, U splitEvent, U polygonCount, U polygonCountLeft, U polygonCountRight) -> PerfectSplitType {
        F min = thrust::get<0>(nodeBbox), max = thrust::get<1>(nodeBbox);
        assert(!(splitPos < min));
        assert(!(max < splitPos));
        if (!(min < max)) {
            return {std::numeric_limits<F>::infinity()};
        }
        F l = splitPos - min;
        F r = max - splitPos;
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
        if ((polygonCountLeft == polygonCount) || (polygonCountRight == polygonCount)) {
            return {std::numeric_limits<F>::infinity()};
        }
        F emptinessFactor(1);
        if (polygonCountLeft == 0) {
            assert(polygonCountRight != 0);
            if (!(min < splitPos)) {
                return {std::numeric_limits<F>::infinity()};
            }
            emptinessFactor = sah.emptinessFactor;
        } else if (polygonCountRight == 0) {
            if (!(splitPos < max)) {
                return {std::numeric_limits<F>::infinity()};
            }
            emptinessFactor = sah.emptinessFactor;
        }
        F x = max - min;
        F y = thrust::get<3>(nodeBbox) - thrust::get<2>(nodeBbox);
        F z = thrust::get<5>(nodeBbox) - thrust::get<4>(nodeBbox);
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
        return {splitCost, splitEvent, splitPos, polygonCountLeft, polygonCountRight};
    };
    auto perfectSplitValueBegin = thrust::make_transform_iterator(perfectSplitInputBegin, thrust::zip_function(toPerfectSplit));
    [[maybe_unused]] auto ends = thrust::reduce_by_key(event.node.cbegin(), event.node.cend(), perfectSplitValueBegin, thrust::make_discard_iterator(), perfectSplitOutputBegin, thrust::equal_to<U>{}, thrust::minimum<PerfectSplitType>{});
    assert(ends.first == thrust::make_discard_iterator(layerNodeOffset.size()));
    timer(" findPerfectSplit reduce_by_key");  // 2.897ms
}
