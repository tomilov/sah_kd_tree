#include <sah_kd_tree/sah_kd_tree.cuh>

void sah_kd_tree::Projection::setNodeCount(U nodeCount)
{
    node.min.resize(nodeCount);
    node.max.resize(nodeCount);
}
