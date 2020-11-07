#include "utility.cuh"

#include <sah_kd_tree/projection.hpp>

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

#include <cassert>

namespace SahKdTree
{
template<I dimension>
void Projection<dimension>::findPerfectSplit(const Params & sah, U nodeCount, const thrust::device_vector<U> & layerNodeOffset, const Y & y, const Z & z)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    {  // calculate l and r polygon count
        event.countLeft.resize(eventCount);
        event.countRight.resize(eventCount);

        auto leftTriangleCountBegin = thrust::make_transform_iterator(event.kind.cbegin(), [] __host__ __device__(I eventKind) -> U { return (eventKind < 0) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.cbegin(), event.node.cend(), leftTriangleCountBegin, event.countLeft.begin());

        auto rightTriangleCountBegin = thrust::make_transform_iterator(event.kind.crbegin(), [] __host__ __device__(I eventKind) -> U { return (0 < eventKind) ? 0 : 1; });
        thrust::exclusive_scan_by_key(event.node.crbegin(), event.node.crend(), rightTriangleCountBegin, event.countRight.rbegin());
    }
    timer(" findPerfectSplit 2 * exclusive_scan_by_key");  // 0.003573

    layer.splitCost.resize(nodeCount);
    layer.splitEvent.resize(nodeCount);
    layer.splitPos.resize(nodeCount);

    layer.polygonCountLeft.resize(nodeCount);
    layer.polygonCountRight.resize(nodeCount);
    timer(" findPerfectSplit resize");  // 0.000667

    auto nodeLimitsBegin = thrust::make_zip_iterator(thrust::make_tuple(node.min.cbegin(), node.max.cbegin(), y.node.min.cbegin(), y.node.max.cbegin(), z.node.min.cbegin(), z.node.max.cbegin()));
    auto nodeBboxBegin = thrust::make_permutation_iterator(nodeLimitsBegin, event.node.cbegin());
    auto splitEventBegin = thrust::make_counting_iterator<U>(0);
    auto perfectSplitInputBegin = thrust::make_zip_iterator(thrust::make_tuple(nodeBboxBegin, event.pos.cbegin(), event.kind.cbegin(), splitEventBegin, event.countLeft.cbegin(), event.countRight.cbegin()));
    using PerfectSplitInputType = IteratorValueType<decltype(perfectSplitInputBegin)>;
    auto perfectSplitBegin = thrust::make_zip_iterator(thrust::make_tuple(layer.splitCost.begin(), layer.splitEvent.begin(), layer.splitPos.begin(), layer.polygonCountLeft.begin(), layer.polygonCountRight.begin()));
    using PerfectSplitType = IteratorValueType<decltype(perfectSplitBegin)>;
    auto toPerfectSplit = [sah] __host__ __device__(PerfectSplitInputType perfectSplitValue) -> PerfectSplitType {
        const auto & nodeBbox = thrust::get<0>(perfectSplitValue);
        F splitPos = thrust::get<1>(perfectSplitValue);
        I eventKind = thrust::get<2>(perfectSplitValue);
        U splitEvent = thrust::get<3>(perfectSplitValue);
        U l = thrust::get<4>(perfectSplitValue), r = thrust::get<5>(perfectSplitValue);  // polygon count
        assert((l != 0) || (r != 0));
        F min = thrust::get<0>(nodeBbox), max = thrust::get<1>(nodeBbox);
        F xl = splitPos - min;
        F xr = max - splitPos;
        if (eventKind < 0) {
            ++splitEvent;
        } else if (eventKind == 0) {
            if ((xl < xr) ? (l != 0) : (r == 0)) {
                ++l;
                ++splitEvent;
            } else {
                ++r;
            }
        }
        F x = max - min;
        F y = thrust::get<3>(nodeBbox) - thrust::get<2>(nodeBbox);
        F z = thrust::get<5>(nodeBbox) - thrust::get<4>(nodeBbox);
        F perimeter = y + z;
        F area = y * z;
        F splitCost = (l * (area + perimeter * xl) + r * (area + perimeter * xr)) / (area + perimeter * x);
        splitCost *= sah.intersectionCost;
        splitCost += sah.traversalCost;
        if ((l == 0) || (r == 0)) {
            splitCost *= sah.emptinessFactor;
        }
        return {splitCost, splitEvent, splitPos, l, r};
    };
    auto perfectSplitValueBegin = thrust::make_transform_iterator(perfectSplitInputBegin, toPerfectSplit);
    [[maybe_unused]] auto ends = thrust::reduce_by_key(event.node.cbegin(), event.node.cend(), perfectSplitValueBegin, thrust::make_discard_iterator(), thrust::make_permutation_iterator(perfectSplitBegin, layerNodeOffset.cbegin()),
                                                       thrust::equal_to<U>{}, thrust::minimum<PerfectSplitType>{});
    assert(ends.first == thrust::make_discard_iterator(layerNodeOffset.size()));
    timer(" findPerfectSplit reduce_by_key");  // 0.003015
}

template void Projection<0>::findPerfectSplit(const Params & sah, U nodeCount, const thrust::device_vector<U> & layerNodeOffset, const Y & y, const Z & z);
template void Projection<1>::findPerfectSplit(const Params & sah, U nodeCount, const thrust::device_vector<U> & layerNodeOffset, const Y & y, const Z & z);
template void Projection<2>::findPerfectSplit(const Params & sah, U nodeCount, const thrust::device_vector<U> & layerNodeOffset, const Y & y, const Z & z);
}  // namespace SahKdTree