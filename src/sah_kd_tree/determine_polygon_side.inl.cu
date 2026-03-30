#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#if 1
#include <thrust/iterator/permutation_iterator.h>
#else
#include <thrust/iterator/transform_output_iterator.h>
#endif
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cassert>

namespace sah_kd_tree
{
template<typename Traits>
template<typename Traits::I dimension>
void Builder<Traits>::determinePolygonSide(const Projection<Traits> & projection)
{
    auto eventBegin = thrust::make_counting_iterator<U>(0);
    auto eventEnd = thrust::make_counting_iterator<U>(projection.event.count);

    auto eventNodes = thrust::raw_pointer_cast(projection.event.node.data());
    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto eventKinds = thrust::raw_pointer_cast(projection.event.kind.data());

    const auto isNotLeftEvent = [eventNodes, nodeSplitDimensions, eventKinds] __host__ __device__(U event) -> bool
    {
        if (nodeSplitDimensions[eventNodes[event]] != dimension) {
            return false;
        }
        return !(0 < eventKinds[event]);
    };
    thrust::scatter_if(exec, eventBegin, eventEnd, projection.event.polygon.cbegin(), eventBegin, polygon.eventRight.begin(), isNotLeftEvent);

    auto polygonRightEvents = thrust::raw_pointer_cast(polygon.eventRight.data());
    auto eventPolygons = thrust::raw_pointer_cast(projection.event.polygon.data());
    auto layerSplitEvents = thrust::raw_pointer_cast(projection.layer.splitEvent.data());
    U layerBase = layer.base;
    const auto toPolygonSide = [polygonRightEvents, eventPolygons, eventNodes, layerBase, layerSplitEvents, eventKinds] __host__ __device__(U eventLeft) -> I
    {
        U eventRight = polygonRightEvents[eventPolygons[eventLeft]];
        assert(!(eventRight < eventLeft));
        U eventNode = eventNodes[eventLeft];
        assert(!(eventNode < layerBase));
        U splitEvent = layerSplitEvents[eventNode - layerBase];
        if (eventRight < splitEvent) {
            return -1;  // goes to left child node
        } else if (eventLeft < splitEvent) {
            assert(eventKinds[eventLeft] != 0);
            static_cast<void>(eventKinds);
            return 0;  // goes to both left child node and right child node (splitted)
        } else {
            return +1;  // goes to right child node
        }
    };
    const auto isNotRightEvent = [eventNodes, nodeSplitDimensions, eventKinds] __host__ __device__(U event) -> bool
    {
        if (nodeSplitDimensions[eventNodes[event]] != dimension) {
            return false;
        }
        return !(eventKinds[event] < 0);
    };
#if 1
    auto polygonSideBegin = thrust::make_permutation_iterator(polygon.side.begin(), projection.event.polygon.cbegin());
    thrust::transform_if(exec, eventBegin, eventEnd, polygonSideBegin, toPolygonSide, isNotRightEvent);
#else
    auto polygonSideBegin = thrust::make_transform_output_iterator(polygon.side.begin(), toPolygonSide);
    thrust::scatter_if(exec, eventBegin, eventEnd, projection.event.polygon.cbegin(), eventBegin, polygonSideBegin, isNotRightEvent);
#endif
}
}  // namespace sah_kd_tree
