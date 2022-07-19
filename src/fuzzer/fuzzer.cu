#include <fuzzer/fuzzer.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <iterator>

namespace fuzzer
{
void testOneInput(const Params & p, const std::vector<Triangle> & t)
{
    sah_kd_tree::Triangle triangle;
    triangle.setTriangle(std::cbegin(t), std::cend(t));

    sah_kd_tree::Projection x, y, z;
    sah_kd_tree::Builder builder;
    sah_kd_tree::linkTriangles(triangle, x, y, z, builder);

    sah_kd_tree::Params params;
    params.emptinessFactor = p.emptinessFactor;
    params.traversalCost = p.traversalCost;
    params.intersectionCost = p.intersectionCost;
    params.maxDepth = p.maxDepth;

    sah_kd_tree::Tree tree = builder(params, x, y, z);
}
}  // namespace fuzzer
