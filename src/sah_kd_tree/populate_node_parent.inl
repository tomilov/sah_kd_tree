#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>

template<typename Traits>
SAH_KD_TREE_INLINE void sah_kd_tree::Builder<Traits>::populateNodeParent()
{
    node.parent.resize(node.count);

    auto parentNodeBegin = thrust::make_counting_iterator<U>(0);
    auto parentNodeEnd = thrust::make_counting_iterator<U>(node.count);
    thrust::scatter_if(parentNodeBegin, parentNodeEnd, node.leftChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), isNotLeaf);
    thrust::scatter_if(parentNodeBegin, parentNodeEnd, node.rightChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), isNotLeaf);
}
