#include "sah_kd_tree/sah_kd_tree.hpp"
#include "sah_kd_tree/utility.cuh"

#include <thrust/advance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void sah_kd_tree::Projection::determinePolygonSide(I dimension, const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto splitDimensionBegin = thrust::make_permutation_iterator(nodeSplitDimension.cbegin(), event.node.cbegin());
    auto eventSideBegin = thrust::make_zip_iterator(thrust::make_tuple(splitDimensionBegin, event.kind.cbegin()));

    auto eventBegin = thrust::make_counting_iterator<U>(0);

    {  // find event counterpart
        auto eventEnd = thrust::next(eventBegin, eventCount);

        auto isNotLeftEvent = [dimension] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
            if (nodeSplitDimension != dimension) {
                return false;
            }
            return !(0 < eventKind);
        };
        thrust::scatter_if(eventBegin, eventEnd, event.polygon.cbegin(), eventSideBegin, polygonEventRight.begin(), thrust::zip_function(isNotLeftEvent));
    }
    timer(" determinePolygonSide scatter_if");  // 1.958ms

    auto isNotRightEvent = [dimension] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
        if (nodeSplitDimension != dimension) {
            return false;
        }
        return !(eventKind < 0);
    };

    auto polygonEventRightBegin = thrust::make_permutation_iterator(polygonEventRight.cbegin(), event.polygon.cbegin());
    auto eventCounterpartBegin = thrust::make_zip_iterator(thrust::make_tuple(eventBegin, polygonEventRightBegin));
    using EventCounterpartType = IteratorValueType<decltype(eventCounterpartBegin)>;
    auto eventNodeOffsetBegin = thrust::make_transform_iterator(event.node.cbegin(), [layerBase] __host__ __device__(U eventNode) -> U {
        assert(!(eventNode < layerBase));
        return eventNode - layerBase;
    });
    auto splitEventBegin = thrust::make_permutation_iterator(layer.splitEvent.cbegin(), eventNodeOffsetBegin);
    auto polygonSideBegin = thrust::make_permutation_iterator(polygonSide.begin(), event.polygon.cbegin());
    auto toPolygonSide = [] __host__ __device__(EventCounterpartType eventBoth, U splitEvent) -> I {
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
    thrust::transform_if(eventCounterpartBegin, thrust::next(eventCounterpartBegin, eventCount), splitEventBegin, eventSideBegin, polygonSideBegin, toPolygonSide, thrust::zip_function(isNotRightEvent));
    timer(" determinePolygonSide transform_if");  // 3.002ms
}
