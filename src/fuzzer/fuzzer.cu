#include <fuzzer/fuzzer.hpp>
#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>

namespace fuzzer
{
void testOneInput(const Params & p, const std::vector<Triangle> & t)
{
    sah_kd_tree::helpers::Triangles triangles;
    triangles.setTriangles(std::cbegin(t), std::cend(t));

    sah_kd_tree::Builder builder;
    sah_kd_tree::helpers::linkTriangles(builder, triangles);

    sah_kd_tree::Params params;
    params.emptinessFactor = p.emptinessFactor;
    params.traversalCost = p.traversalCost;
    params.intersectionCost = p.intersectionCost;
    params.maxDepth = p.maxDepth;

    sah_kd_tree::Tree tree = builder(params);
}
}  // namespace fuzzer
