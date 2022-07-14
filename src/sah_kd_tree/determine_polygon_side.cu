#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/type_traits.cuh>

#include <thrust/iterator/counting_iterator.h>
#if 1
#include <thrust/iterator/permutation_iterator.h>
#else
#include <thrust/iterator/transform_output_iterator.h>
#endif
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cassert>

namespace sah_kd_tree
{
template<I dimension>
void Builder::determinePolygonSide(const Projection & projection)
{
    auto eventBegin = thrust::make_counting_iterator<U>(0);
    auto eventEnd = thrust::make_counting_iterator<U>(projection.event.count);

    auto eventNodes = projection.event.node.data().get();
    auto nodeSplitDimensions = node.splitDimension.data().get();
    auto eventKinds = projection.event.kind.data().get();

    const auto isNotLeftEvent = [eventNodes, nodeSplitDimensions, eventKinds] __host__ __device__(U event) -> bool {
        if (nodeSplitDimensions[eventNodes[event]] != dimension) {
            return false;
        }
        return !(0 < eventKinds[event]);
    };
    thrust::scatter_if(eventBegin, eventEnd, projection.event.polygon.cbegin(), eventBegin, polygon.eventRight.begin(), isNotLeftEvent);

    auto polygonRightEvents = polygon.eventRight.data().get();
    auto eventPolygons = projection.event.polygon.data().get();
    auto layerSplitEvents = projection.layer.splitEvent.data().get();
    U layerBase = layer.base;
    const auto toPolygonSide = [polygonRightEvents, eventPolygons, eventNodes, layerBase, layerSplitEvents, eventKinds] __host__ __device__(U eventLeft) -> I {
        U eventRight = polygonRightEvents[eventPolygons[eventLeft]];
        assert(!(eventRight < eventLeft));
        U eventNode = eventNodes[eventLeft];
        assert(!(eventNode < layerBase));
        U splitEvent = layerSplitEvents[eventNode - layerBase];
        if (eventRight < splitEvent) {
            return -1;  // goes to left child node
        } else if (eventLeft < splitEvent) {
            assert(eventKinds[eventLeft] != 0);
            return 0;  // goes to both left child node and right child node (splitted)
        } else {
            return +1;  // goes to right child node
        }
    };
    const auto isNotRightEvent = [eventNodes, nodeSplitDimensions, eventKinds] __host__ __device__(U event) -> bool {
        if (nodeSplitDimensions[eventNodes[event]] != dimension) {
            return false;
        }
        return !(eventKinds[event] < 0);
    };
#if 1
    auto polygonSideBegin = thrust::make_permutation_iterator(polygon.side.begin(), projection.event.polygon.cbegin());
    thrust::transform_if(eventBegin, eventEnd, polygonSideBegin, toPolygonSide, isNotRightEvent);
#else
    auto polygonSideBegin = thrust::make_transform_output_iterator(polygon.side.begin(), toPolygonSide);
    thrust::scatter_if(eventBegin, eventEnd, projection.event.polygon.cbegin(), eventBegin, polygonSideBegin, isNotRightEvent);
#endif
}

template void Builder::determinePolygonSide<0>(const Projection & x) SAH_KD_TREE_EXPORT;
template void Builder::determinePolygonSide<1>(const Projection & y) SAH_KD_TREE_EXPORT;
template void Builder::determinePolygonSide<2>(const Projection & z) SAH_KD_TREE_EXPORT;
}  // namespace sah_kd_tree
