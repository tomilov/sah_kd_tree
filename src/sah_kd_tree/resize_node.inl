#include <sah_kd_tree/sah_kd_tree.cuh>

//#include <limits>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::resizeNode()
{
    //node.splitCost.resize(node.count, std::numeric_limits<F>::quiet_NaN());
    node.splitDimension.resize(node.count, kNoSplitDimension);
    node.splitPos.resize(node.count);
    node.leftChild.resize(node.count);
    node.rightChild.resize(node.count);
    node.polygonCountLeft.resize(node.count);
    node.polygonCountRight.resize(node.count);
}
