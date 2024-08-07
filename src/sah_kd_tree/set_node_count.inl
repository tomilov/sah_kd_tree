#include <sah_kd_tree/sah_kd_tree.cuh>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::setNodeCount(Projection<Traits> & x, Projection<Traits> & y, Projection<Traits> & z) const
{
    x.node.min.resize(node.count);
    x.node.max.resize(node.count);

    y.node.min.resize(node.count);
    y.node.max.resize(node.count);

    z.node.min.resize(node.count);
    z.node.max.resize(node.count);
}
