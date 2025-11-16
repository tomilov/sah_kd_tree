#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/memory.h>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::updateSplittedPolygonCount()
{
    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto nodePolygonCountLefts = thrust::raw_pointer_cast(node.polygonCountLeft.data());
    auto nodePolygonCountRights = thrust::raw_pointer_cast(node.polygonCountRight.data());
    auto nodePolygonCounts = thrust::raw_pointer_cast(node.polygonCount.data());
    const auto toSplittedPolygonCount = [nodeSplitDimensions, nodePolygonCountLefts, nodePolygonCountRights, nodePolygonCounts] __host__ __device__(U layerNode) -> U {
        if (nodeSplitDimensions[layerNode] < 0) {
            return 0;
        }
        U polygonCountLeft = nodePolygonCountLefts[layerNode];
        U polygonCountRight = nodePolygonCountRights[layerNode];
        U polygonCount = nodePolygonCounts[layerNode];
        assert(!(polygonCountLeft + polygonCountRight < polygonCount));
        return polygonCountLeft + polygonCountRight - polygonCount;
    };
    auto layerNodeBegin = thrust::make_counting_iterator<U>(layer.base);
    auto layerNodeEnd = thrust::next(layerNodeBegin, layer.size);
    polygon.splittedCount = thrust::transform_reduce(layerNodeBegin, layerNodeEnd, toSplittedPolygonCount, static_cast<U>(0), cuda::std::plus<U>{});
}
