#include <fuzzer/fuzzer.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <iterator>

namespace fuzzer
{
void testOneInput(const Params & p, const std::vector<Triangle> & t)
{
    using Traits = sah_kd_tree::DefaultTraits;

    sah_kd_tree::Triangle<Traits> triangle;
    triangle.setTriangle(std::cbegin(t), std::cend(t));

    sah_kd_tree::Builder<Traits> builder;
    sah_kd_tree::Projection<Traits> x, y, z;
    sah_kd_tree::linkTriangles(triangle, x, y, z, builder);

    sah_kd_tree::Params<Traits> params;
    params.emptinessFactor = p.emptinessFactor;
    params.traversalCost = p.traversalCost;
    params.intersectionCost = p.intersectionCost;
    params.maxDepth = p.maxDepth;

    const typename Traits::Cancel cancel = []
    {
        return false;
    };
    sah_kd_tree::Tree<Traits> tree = builder(cancel, params, x, y, z).value();
}
}  // namespace fuzzer
