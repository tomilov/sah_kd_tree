#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cassert>

void sah_kd_tree::Builder::separateSplittedPolygon()
{
    polygon.triangle.resize(polygon.count + polygon.splittedCount);
    polygon.node.resize(polygon.count + polygon.splittedCount);
    splittedPolygon.resize(polygon.splittedCount);

    auto polygonBegin = thrust::make_counting_iterator<U>(0);

    auto polygonNodes = polygon.node.data().get();
    auto nodeSplitDimensions = node.splitDimension.data().get();
    auto polygonSides = polygon.side.data().get();
    U layerBase = layer.base;
    const auto isSplittedPolygon = [layerBase, polygonNodes, nodeSplitDimensions, polygonSides] __host__ __device__(U polygon) -> bool
    {
        U polygonNode = polygonNodes[polygon];
        if (polygonNode < layerBase) {
            return false;
        }
        if (nodeSplitDimensions[polygonNode] < 0) {
            return false;
        }
        return polygonSides[polygon] == 0;
    };
    auto polygonTriangles = polygon.triangle.data().get();
    auto polygonTriangleAndNodeBegin = thrust::make_zip_iterator(polygon.triangle.begin(), polygon.node.begin());
    auto splittedPolygonOutputBegin = thrust::make_zip_iterator(splittedPolygon.begin(), thrust::next(polygonTriangleAndNodeBegin, polygon.count));
    using SplittedPolygonType = thrust::iterator_value_t<decltype(splittedPolygonOutputBegin)>;
    const auto toSplittedPolygon = [polygonTriangles, polygonNodes] __host__ __device__(U polygon) -> SplittedPolygonType
    {
        return {polygon, {polygonTriangles[polygon], polygonNodes[polygon]}};
    };
    auto splittedPolygonInputBegin = thrust::make_transform_iterator(polygonBegin, toSplittedPolygon);
    auto splittedPolygonInputEnd = thrust::next(splittedPolygonInputBegin, polygon.count);
    [[maybe_unused]] auto splittedPolygonOutputEnd = thrust::copy_if(splittedPolygonInputBegin, splittedPolygonInputEnd, polygonBegin, splittedPolygonOutputBegin, isSplittedPolygon);
    assert(thrust::next(splittedPolygonOutputBegin, polygon.splittedCount) == splittedPolygonOutputEnd);
}
