#include "utility.cuh"

#include <sah_kd_tree/builder.hpp>

#include <thrust/sequence.h>

#include <cassert>

namespace SahKdTree
{
SahKdTree Builder::operator()(const Params & /*sah*/)
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
    node.polygonCount.node.assign(1, triangleCount);
    node.polygonCount.l.resize(1);
    node.polygonCount.r.resize(1);

    return {};
}
}  // namespace SahKdTree
