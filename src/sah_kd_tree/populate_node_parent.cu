#include <sah_kd_tree/sah_kd_tree.cuh>

void sah_kd_tree::Builder::populateNodeParent()
{
    node.parent.resize(node.count);
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), node.leftChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), isNotLeaf);
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), node.rightChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), isNotLeaf);
}
