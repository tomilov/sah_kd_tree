#include "utility.cuh"

#include <SahKdTree.hpp>

void SahKdTree::Projection::setNodeCount(U nodeCount)
{
    node.min.resize(nodeCount);
    node.max.resize(nodeCount);
}
