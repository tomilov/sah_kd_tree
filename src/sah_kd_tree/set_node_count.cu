#include "sah_kd_tree/sah_kd_tree.hpp"
#include "sah_kd_tree/utility.cuh"

void sah_kd_tree::Projection::setNodeCount(U nodeCount)
{
    node.min.resize(nodeCount);
    node.max.resize(nodeCount);
}
