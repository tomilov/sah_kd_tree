#include "utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void SahKdTree::Projection::determinePolygonSide(I dimension, const thrust::device_vector<I> & nodeSplitDimension, U baseNode, thrust::device_vector<U> & polygonEventLeft, thrust::device_vector<U> & polygonEventRight,
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

        auto isNotRightEvent = [dimension] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
            if (nodeSplitDimension != dimension) {
                return false;
            }
            return !(eventKind < 0);
        };
        thrust::scatter_if(eventBegin, eventEnd, event.polygon.cbegin(), eventSideBegin, polygonEventLeft.begin(), thrust::zip_function(isNotRightEvent));

        // TODO: optimize out one of (polygonEventLeft, polygonEventRight)
        auto isNotLeftEvent = [dimension] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
            if (nodeSplitDimension != dimension) {
                return false;
            }
            return !(0 < eventKind);
        };
        thrust::scatter_if(eventBegin, eventEnd, event.polygon.cbegin(), eventSideBegin, polygonEventRight.begin(), thrust::zip_function(isNotLeftEvent));
    }
    timer(" determinePolygonSide 2 * scatter_if");  // 1.958ms

    auto polygonEventBothBegin = thrust::make_zip_iterator(thrust::make_tuple(polygonEventLeft.cbegin(), polygonEventRight.cbegin()));
    auto eventBothBegin = thrust::make_permutation_iterator(polygonEventBothBegin, event.polygon.cbegin());
    using EventBothType = IteratorValueType<decltype(eventBothBegin)>;
    auto splitEventBegin = thrust::make_permutation_iterator(thrust::prev(layer.splitEvent.cbegin(), baseNode), event.node.cbegin());
    auto polygonSideBegin = thrust::make_permutation_iterator(polygonSide.begin(), event.polygon.cbegin());
    auto toPolygonSide = [] __host__ __device__(EventBothType eventBoth, U splitEvent) -> I {
        U eventLeft = thrust::get<0>(eventBoth);
        U eventRight = thrust::get<1>(eventBoth);
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
    thrust::transform_if(eventBothBegin, thrust::next(eventBothBegin, eventCount), splitEventBegin, splitDimensionBegin, polygonSideBegin, toPolygonSide, isX);
    timer(" determinePolygonSide transform_if");  // 3.002ms
}
