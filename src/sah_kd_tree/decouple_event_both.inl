#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cassert>

void sah_kd_tree::Projection::decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide)
{
    auto eventBegin = thrust::make_counting_iterator<U>(0);
    auto eventEnd = thrust::make_counting_iterator<U>(event.count);

    auto eventNodes = event.node.data().get();
    auto nodeSplitDimensions = nodeSplitDimension.data().get();
    auto eventPolygons = event.polygon.data().get();
    auto polygonSides = polygonSide.data().get();

    auto & eventLeft = event.polygonCountLeft;
    assert(!(eventLeft.size() < event.count));
    const auto isLeftPolygon = [eventNodes, nodeSplitDimensions, eventPolygons, polygonSides] __host__ __device__(U event) -> bool {
        if (nodeSplitDimensions[eventNodes[event]] < 0) {
            return false;
        }
        return polygonSides[eventPolygons[event]] < 0;
    };
    auto eventLeftEnd = thrust::copy_if(eventBegin, eventEnd, eventLeft.begin(), isLeftPolygon);
    eventLeft.erase(eventLeftEnd, eventLeft.end());

    auto & eventRight = event.polygonCountRight;
    assert(!(eventRight.size() < event.count));
    const auto isRightPolygon = [eventNodes, nodeSplitDimensions, eventPolygons, polygonSides] __host__ __device__(U event) -> bool {
        if (nodeSplitDimensions[eventNodes[event]] < 0) {
            return false;
        }
        return 0 < polygonSides[eventPolygons[event]];
    };
    auto eventRightEnd = thrust::copy_if(eventBegin, eventEnd, eventRight.begin(), isRightPolygon);
    eventRight.erase(eventRightEnd, eventRight.end());
}
