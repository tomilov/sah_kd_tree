#include "utility.cuh"

#include <sah_kd_tree/builder.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <cassert>

namespace SahKdTree
{
SahKdTree Builder::operator()(const Params & sah)
{
    auto triangleCount = U(x.triangle.a.size());
    assert(triangleCount == U(y.triangle.a.size()));
    assert(triangleCount == U(z.triangle.a.size()));

    Timer timer;

    x.calculateTriangleBbox();
    y.calculateTriangleBbox();
    z.calculateTriangleBbox();
    timer("calculateTriangleBbox");  // 0.004484

    x.caluculateRootNodeBbox();
    y.caluculateRootNodeBbox();
    z.caluculateRootNodeBbox();
    timer("caluculateRootNodeBbox");  // 0.001709

    x.generateInitialEvent();
    y.generateInitialEvent();
    z.generateInitialEvent();
    timer("generateInitialEvent");  // 0.138127

    polygon.triangle.resize(triangleCount);
    thrust::sequence(polygon.triangle.begin(), polygon.triangle.end());
    polygon.node.assign(triangleCount, U(0));

    node.splitDimension.resize(1);
    node.splitPos.resize(1);
    node.l.resize(1);
    node.r.resize(1);
    node.polygonCount.assign(1, triangleCount);
    node.polygonCountLeft.resize(1);
    node.polygonCountRight.resize(1);
    timer("init builder");  // 0.003133

    // layer
    U baseNode = 0;
    U nodeCount = 1;

    for (U depth = 0; depth < sah.maxDepth; ++depth) {
        {
            layerNodeOffset.resize(nodeCount);

            auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
            auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
            auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, nodeCount), thrust::next(node.polygonCount.cbegin(), baseNode), layerNodeOffset.begin(), isNodeNotEmpty);
            layerNodeOffset.erase(layerNodeEnd, layerNodeOffset.end());
            timer("layerNodeOffset");
        }

        x.findPerfectSplit(sah, nodeCount, layerNodeOffset, y, z);
        y.findPerfectSplit(sah, nodeCount, layerNodeOffset, z, x);
        z.findPerfectSplit(sah, nodeCount, layerNodeOffset, x, y);
        break;
    }
    return {};
}
}  // namespace SahKdTree
