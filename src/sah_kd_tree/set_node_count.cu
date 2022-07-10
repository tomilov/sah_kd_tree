#include <sah_kd_tree/sah_kd_tree.cuh>

void sah_kd_tree::Builder::setNodeCount(Projection & x, Projection & y, Projection & z)
{
    x.node.min.resize(node.count);
    x.node.max.resize(node.count);

    y.node.min.resize(node.count);
    y.node.max.resize(node.count);

    z.node.min.resize(node.count);
    z.node.max.resize(node.count);
}
