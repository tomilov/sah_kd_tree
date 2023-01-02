#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

void sah_kd_tree::Builder::updateSplittedPolygonCount()
{
    auto nodeSplitDimensions = node.splitDimension.data().get();
    auto nodePolygonCountLefts = node.polygonCountLeft.data().get();
    auto nodePolygonCountRights = node.polygonCountRight.data().get();
    auto nodePolygonCounts = node.polygonCount.data().get();
    const auto toSplittedPolygonCount = [nodeSplitDimensions, nodePolygonCountLefts, nodePolygonCountRights, nodePolygonCounts] __host__ __device__(U layerNode) -> U
    {
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
    polygon.splittedCount = thrust::transform_reduce(layerNodeBegin, layerNodeEnd, toSplittedPolygonCount, U(0), thrust::plus<U>{});
}
