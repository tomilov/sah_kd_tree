#include <sah_kd_tree/sah_kd_tree.cuh>

namespace sah_kd_tree
{
void Builder::resizeNode()
{
    node.splitDimension.resize(node.count, I(-1));
    node.splitPos.resize(node.count);
    node.leftChild.resize(node.count);
    node.rightChild.resize(node.count);
    node.polygonCountLeft.resize(node.count);
    node.polygonCountRight.resize(node.count);
}
}  // namespace sah_kd_tree
