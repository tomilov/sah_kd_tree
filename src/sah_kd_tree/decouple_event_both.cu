#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

#include <cassert>

void sah_kd_tree::Projection::decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide)
{
    auto eventBegin = thrust::make_counting_iterator<U>(0);

    auto splitDimensionBegin = thrust::make_permutation_iterator(nodeSplitDimension.cbegin(), event.node.cbegin());
    auto sideBegin = thrust::make_permutation_iterator(polygonSide.cbegin(), event.polygon.cbegin());
    auto stencilBegin = thrust::make_zip_iterator(splitDimensionBegin, sideBegin);

    auto & eventLeft = event.polygonCountLeft;
    assert(!(eventLeft.size() < event.count));
    const auto isLeftPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return polygonSide < 0;
    };
    auto eventLeftEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, event.count), stencilBegin, eventLeft.begin(), thrust::make_zip_function(isLeftPolygon));
    eventLeft.erase(eventLeftEnd, eventLeft.end());

    auto & eventRight = event.polygonCountRight;
    assert(!(eventRight.size() < event.count));
    const auto isRightPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return 0 < polygonSide;
    };
    auto eventRightEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, event.count), stencilBegin, eventRight.begin(), thrust::make_zip_function(isRightPolygon));
    eventRight.erase(eventRightEnd, eventRight.end());
}
