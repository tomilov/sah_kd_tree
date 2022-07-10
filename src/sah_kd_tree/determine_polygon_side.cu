#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/type_traits.cuh>

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

namespace sah_kd_tree
{
template<I dimension>
void Builder::determinePolygonSide(const Projection & projection)
{
    auto splitDimensionBegin = thrust::make_permutation_iterator(node.splitDimension.cbegin(), projection.event.node.cbegin());
    auto eventSideBegin = thrust::make_zip_iterator(splitDimensionBegin, projection.event.kind.cbegin());

    auto eventBegin = thrust::make_counting_iterator<U>(0);

    {  // find event counterpart
        auto eventEnd = thrust::next(eventBegin, projection.event.count);

        const auto isNotLeftEvent = [] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
            if (nodeSplitDimension != dimension) {
                return false;
            }
            return !(0 < eventKind);
        };
        thrust::scatter_if(eventBegin, eventEnd, projection.event.polygon.cbegin(), eventSideBegin, polygon.eventRight.begin(), thrust::make_zip_function(isNotLeftEvent));
    }

    const auto isNotRightEvent = [] __host__ __device__(I nodeSplitDimension, I eventKind) -> bool {
        if (nodeSplitDimension != dimension) {
            return false;
        }
        return !(eventKind < 0);
    };

    auto polygonEventRightBegin = thrust::make_permutation_iterator(polygon.eventRight.cbegin(), projection.event.polygon.cbegin());
    auto eventCounterpartBegin = thrust::make_zip_iterator(eventBegin, polygonEventRightBegin);
    using EventCounterpartType = IteratorValueType<decltype(eventCounterpartBegin)>;
    U layerBase = layer.base;
    const auto eventNodeOffsetBegin = thrust::make_transform_iterator(projection.event.node.cbegin(), [layerBase] __host__ __device__(U eventNode) -> U {
        assert(!(eventNode < layerBase));
        return eventNode - layerBase;
    });
    auto splitEventBegin = thrust::make_permutation_iterator(projection.layer.splitEvent.cbegin(), eventNodeOffsetBegin);
    auto polygonSideBegin = thrust::make_permutation_iterator(polygon.side.begin(), projection.event.polygon.cbegin());
    const auto toPolygonSide = [] __host__ __device__(EventCounterpartType eventBoth, U splitEvent) -> I {
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
    thrust::transform_if(eventCounterpartBegin, thrust::next(eventCounterpartBegin, projection.event.count), splitEventBegin, eventSideBegin, polygonSideBegin, toPolygonSide, thrust::make_zip_function(isNotRightEvent));
}

template void Builder::determinePolygonSide<0>(const Projection & x);
template void Builder::determinePolygonSide<1>(const Projection & y);
template void Builder::determinePolygonSide<2>(const Projection & z);
}  // namespace sah_kd_tree
