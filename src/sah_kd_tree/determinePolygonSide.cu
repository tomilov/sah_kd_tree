#include "utility.cuh"

#include <sah_kd_tree/projection.hpp>

#include <thrust/advance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

void SahKdTree::Projection::determinePolygonSide(I dimension, const thrust::device_vector<I> & nodeSplitDimension, U baseNode, thrust::device_vector<U> & polygonLeftEvent, thrust::device_vector<U> & polygonRightEvent,
                                                 thrust::device_vector<I> & polygonSide)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto splitDimensionBegin = thrust::make_permutation_iterator(nodeSplitDimension.cbegin(), event.node.cbegin());

    {  // find event counterpart
        auto eventBegin = thrust::make_counting_iterator<U>(0);
        auto eventEnd = thrust::next(eventBegin, eventCount);

        auto eventSideBegin = thrust::make_zip_iterator(thrust::make_tuple(splitDimensionBegin, event.kind.cbegin()));
        using EventSideType = IteratorValueType<decltype(eventSideBegin)>;

        auto isNotRightEvent = [dimension] __host__ __device__(EventSideType eventSide) -> bool {
            I nodeSplitDimension = thrust::get<0>(eventSide);
            if (nodeSplitDimension != dimension) {
                return false;
            }
            I eventKind = thrust::get<1>(eventSide);
            return !(eventKind < 0);
        };
        thrust::scatter_if(eventBegin, eventEnd, event.polygon.cbegin(), eventSideBegin, polygonLeftEvent.begin(), isNotRightEvent);

        // TODO: optimize out one of (polygonLeftEvent, polygonRightEvent)
        auto isNotLeftEvent = [dimension] __host__ __device__(EventSideType eventSide) -> bool {
            I nodeSplitDimension = thrust::get<0>(eventSide);
            if (nodeSplitDimension != dimension) {
                return false;
            }
            I eventKind = thrust::get<1>(eventSide);
            return !(0 < eventKind);
        };
        thrust::scatter_if(eventBegin, eventEnd, event.polygon.cbegin(), eventSideBegin, polygonRightEvent.begin(), isNotLeftEvent);
    }
    timer(" determinePolygonSide 2 * scatter_if");  // 0.002360

    auto splitEventBegin = thrust::make_permutation_iterator(thrust::prev(layer.splitEvent.cbegin(), baseNode), event.node.cbegin());
    auto polygonEventLeftRightBegin = thrust::make_zip_iterator(thrust::make_tuple(polygonLeftEvent.cbegin(), polygonRightEvent.cbegin()));
    auto eventLeftRightBegin = thrust::make_permutation_iterator(polygonEventLeftRightBegin, event.polygon.cbegin());
    using EventLeftRightType = IteratorValueType<decltype(eventLeftRightBegin)>;
    auto polygonSideBegin = thrust::make_permutation_iterator(polygonSide.begin(), event.polygon.cbegin());
    auto toPolygonSide = [] __host__ __device__(EventLeftRightType eventLeftRight, U splitEvent) -> I {
        U eventLeft = thrust::get<0>(eventLeftRight);
        U eventRight = thrust::get<1>(eventLeftRight);
        assert(!(eventRight < eventLeft));
        if (eventRight < splitEvent) {
            return -1;  // goes to left child node
        } else if (eventLeft < splitEvent) {
            return 0;  // goes to both left child node and right child node (splitted), assert(eventKind != 0)
        } else {
            return +1;  // goes to right child node
        }
    };
    auto isX = [dimension] __host__ __device__(I nodeSplitDimension) -> bool { return nodeSplitDimension == dimension; };
    thrust::transform_if(eventLeftRightBegin, thrust::next(eventLeftRightBegin, eventCount), splitEventBegin, splitDimensionBegin, polygonSideBegin, toPolygonSide, isX);
    timer(" determinePolygonSide transform_if");  // 0.003103
}
