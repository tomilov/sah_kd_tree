#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cassert>

namespace sah_kd_tree
{
template<I dimension>
void Builder::calculateRope(Direction direction, const Projection & y, const Projection & z, thrust::device_vector<U> & nodeRightRope)
{
    nodeRightRope.resize(node.count);

    auto yMins = y.node.min.data().get();
    auto yMaxs = y.node.max.data().get();
    auto zMins = z.node.min.data().get();
    auto zMaxs = z.node.max.data().get();
    auto parentNodes = node.parentNode.data().get();
    auto leftChildren = node.leftChild.data().get();
    auto rightChildren = node.rightChild.data().get();
    auto splitDimensions = node.splitDimension.data().get();
    auto splitPositions = node.splitPos.data().get();
    const auto getRightRope = [yMins, yMaxs, zMins, zMaxs, parentNodes, direction, leftChildren, rightChildren, splitDimensions, splitPositions] __host__ __device__(U node) -> U {
        U siblingNode = node;
        for (;;) {
            if (siblingNode == 0) {
                return 0;  // ray miss
            }
            U parentNode = parentNodes[siblingNode];
            if (splitDimensions[parentNode] == dimension) {
                if (siblingNode == ((direction == Direction::kPositive) ? leftChildren : rightChildren)[parentNode]) {
                    if (siblingNode == node) {
                        return ((direction == Direction::kNegative) ? leftChildren : rightChildren)[parentNode];
                    }
                    siblingNode = ((direction == Direction::kNegative) ? leftChildren : rightChildren)[parentNode];
                    break;
                }
            }
            siblingNode = parentNode;
        }
        F yMin = yMins[node];
        F yMax = yMaxs[node];
        F zMin = zMins[node];
        F zMax = zMaxs[node];
        for (;;) {
            I siblingSplitDimension = splitDimensions[siblingNode];
            if (siblingSplitDimension < 0) {
                break;
            }
            assert(!(yMin < yMins[siblingNode]));
            assert(!(yMaxs[siblingNode] < yMax));
            assert(!(zMin < zMins[siblingNode]));
            assert(!(zMaxs[siblingNode] < zMax));
            if (siblingSplitDimension == dimension) {
                siblingNode = ((direction == Direction::kPositive) ? leftChildren : rightChildren)[siblingNode];
            } else if (siblingSplitDimension == ((dimension + 1) % 3)) {
                F siblingSplitPosition = splitPositions[siblingNode];
                if (!(siblingSplitPosition < yMax)) {
                    siblingNode = leftChildren[siblingNode];
                } else if (!(yMin < siblingSplitPosition)) {
                    siblingNode = rightChildren[siblingNode];
                } else {
                    break;
                }
            } else if (siblingSplitDimension == ((dimension + 2) % 3)) {
                F siblingSplitPosition = splitPositions[siblingNode];
                if (!(siblingSplitPosition < zMax)) {
                    siblingNode = leftChildren[siblingNode];
                } else if (!(zMin < siblingSplitPosition)) {
                    siblingNode = rightChildren[siblingNode];
                } else {
                    break;
                }
            }
        }
        return siblingNode;
    };
    thrust::transform(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), nodeRightRope.begin(), getRightRope);
}

template void Builder::calculateRope<0>(Direction direction, const Projection & y, const Projection & z, thrust::device_vector<U> & nodeRightRope);
template void Builder::calculateRope<1>(Direction direction, const Projection & y, const Projection & z, thrust::device_vector<U> & nodeRightRope);
template void Builder::calculateRope<2>(Direction direction, const Projection & y, const Projection & z, thrust::device_vector<U> & nodeRightRope);
}  // namespace sah_kd_tree
