#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Projection<Traits>::decoupleEventBoth(const Vector<I> & nodeSplitDimension, const Vector<I> & polygonSide)
{
    auto eventBegin = thrust::make_counting_iterator<U>(0);
    auto eventEnd = thrust::make_counting_iterator<U>(event.count);

    auto eventNodes = thrust::raw_pointer_cast(event.node.data());
    auto nodeSplitDimensions = thrust::raw_pointer_cast(nodeSplitDimension.data());
    auto eventPolygons = thrust::raw_pointer_cast(event.polygon.data());
    auto polygonSides = thrust::raw_pointer_cast(polygonSide.data());

    auto & eventLeft = event.polygonCountLeft;
    assert(!(eventLeft.size() < event.count));
    const auto isLeftPolygon = [eventNodes, nodeSplitDimensions, eventPolygons, polygonSides] __host__ __device__(U event) -> bool
    {
        if (nodeSplitDimensions[eventNodes[event]] < 0) {
            return false;
        }
        return polygonSides[eventPolygons[event]] < 0;
    };
    auto eventLeftEnd = thrust::copy_if(exec, eventBegin, eventEnd, eventLeft.begin(), isLeftPolygon);
    eventLeft.erase(eventLeftEnd, eventLeft.end());

    auto & eventRight = event.polygonCountRight;
    assert(!(eventRight.size() < event.count));
    const auto isRightPolygon = [eventNodes, nodeSplitDimensions, eventPolygons, polygonSides] __host__ __device__(U event) -> bool
    {
        if (nodeSplitDimensions[eventNodes[event]] < 0) {
            return false;
        }
        return 0 < polygonSides[eventPolygons[event]];
    };
    auto eventRightEnd = thrust::copy_if(exec, eventBegin, eventEnd, eventRight.begin(), isRightPolygon);
    eventRight.erase(eventRightEnd, eventRight.end());
}
