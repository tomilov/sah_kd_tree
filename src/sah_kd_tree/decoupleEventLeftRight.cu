#include "utility.cuh"

#include <sah_kd_tree/projection.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/zip_function.h>

void SahKdTree::Projection::decoupleEventLeftRight(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto eventBegin = thrust::make_counting_iterator<U>(0);

    auto splitDimensionBegin = thrust::make_permutation_iterator(nodeSplitDimension.cbegin(), event.node.cbegin());
    auto polygonSideBegin = thrust::make_permutation_iterator(polygonSide.cbegin(), event.polygon.cbegin());
    auto polygonStencilBegin = thrust::make_zip_iterator(thrust::make_tuple(splitDimensionBegin, polygonSideBegin));

    auto isLeftPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return polygonSide < 0;
    };
    auto eventLeftEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, eventCount), polygonStencilBegin, event.countLeft.begin(), thrust::make_zip_function(isLeftPolygon));
    event.countLeft.erase(eventLeftEnd, event.countLeft.end());
    timer(" decoupleEventLeftRight copy reft");  // 1.033ms

    auto isRightPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return 0 < polygonSide;
    };
    auto eventRightEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, eventCount), polygonStencilBegin, event.countRight.begin(), thrust::make_zip_function(isRightPolygon));
    event.countRight.erase(eventRightEnd, event.countRight.end());
    timer(" decoupleEventLeftRight copy right");  // 1.503ms
}
