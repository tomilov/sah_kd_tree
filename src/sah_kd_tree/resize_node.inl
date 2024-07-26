#include <sah_kd_tree/sah_kd_tree.cuh>

template<typename MemoryResource>
SAH_KD_TREE_INLINE void sah_kd_tree::Builder<MemoryResource>::resizeNode()
{
    node.splitDimension.resize(node.count, I(-1));
    node.splitPos.resize(node.count);
    node.leftChild.resize(node.count);
    node.rightChild.resize(node.count);
    node.polygonCountLeft.resize(node.count);
    node.polygonCountRight.resize(node.count);
}
