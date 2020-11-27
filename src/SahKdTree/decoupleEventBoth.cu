#include "Utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

void SahKdTree::Projection::decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto eventBegin = thrust::make_counting_iterator<U>(0);

    auto splitDimensionBegin = thrust::make_permutation_iterator(nodeSplitDimension.cbegin(), event.node.cbegin());
    auto polygonSideBegin = thrust::make_permutation_iterator(polygonSide.cbegin(), event.polygon.cbegin());
    auto polygonStencilBegin = thrust::make_zip_iterator(thrust::make_tuple(splitDimensionBegin, polygonSideBegin));

    auto & eventLeft = event.polygonCountLeft;
    assert(!(eventLeft.size() < eventCount));
    auto isLeftPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return polygonSide < 0;
    };
    auto eventLeftEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, eventCount), polygonStencilBegin, eventLeft.begin(), thrust::make_zip_function(isLeftPolygon));
    eventLeft.erase(eventLeftEnd, eventLeft.end());
    timer(" decoupleEventBoth copy reft");  // 1.033ms

    auto & eventRight = event.polygonCountRight;
    assert(!(eventRight.size() < eventCount));
    auto isRightPolygon = [] __host__ __device__(I splitDimension, I polygonSide) -> bool {
        if (splitDimension < 0) {
            return false;
        }
        return 0 < polygonSide;
    };
    auto eventRightEnd = thrust::copy_if(eventBegin, thrust::next(eventBegin, eventCount), polygonStencilBegin, eventRight.begin(), thrust::make_zip_function(isRightPolygon));
    eventRight.erase(eventRightEnd, eventRight.end());
    timer(" decoupleEventBoth copy right");  // 1.503ms
}
