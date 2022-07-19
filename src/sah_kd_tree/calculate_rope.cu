#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cassert>

namespace sah_kd_tree
{
template<I dimension, bool forth>
void Builder::calculateRope(Projection & x, const Projection & y, const Projection & z) const
{
    auto & nodeRope = forth ? x.node.rightRope : x.node.leftRope;
    nodeRope.resize(node.count);

    auto yMins = y.node.min.data().get();
    auto yMaxs = y.node.max.data().get();
    auto zMins = z.node.min.data().get();
    auto zMaxs = z.node.max.data().get();
    auto parents = node.parent.data().get();
    auto leftChildren = node.leftChild.data().get();
    auto rightChildren = node.rightChild.data().get();
    auto splitDimensions = node.splitDimension.data().get();
    auto splitPositions = node.splitPos.data().get();
    const auto getRightRope = [yMins, yMaxs, zMins, zMaxs, parents, leftChildren, rightChildren, splitDimensions, splitPositions] __host__ __device__(U node) -> U {
        U siblingNode = node;
        for (;;) {
            if (siblingNode == 0) {
                return 0;  // ray miss
            }
            U parent = parents[siblingNode];
            if (splitDimensions[parent] == dimension) {
                if (siblingNode == (forth ? leftChildren : rightChildren)[parent]) {
                    if (siblingNode == node) {
                        return (forth ? rightChildren : leftChildren)[parent];
                    }
                    siblingNode = (forth ? rightChildren : leftChildren)[parent];
                    break;
                }
            }
            siblingNode = parent;
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
                siblingNode = (forth ? leftChildren : rightChildren)[siblingNode];
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
    thrust::transform(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), nodeRope.begin(), getRightRope);
}

template void Builder::calculateRope<0, false>(Projection & x, const Projection & y, const Projection & z) const;
template void Builder::calculateRope<0, true>(Projection & x, const Projection & y, const Projection & z) const;
template void Builder::calculateRope<1, false>(Projection & y, const Projection & z, const Projection & x) const;
template void Builder::calculateRope<1, true>(Projection & y, const Projection & z, const Projection & x) const;
template void Builder::calculateRope<2, false>(Projection & z, const Projection & x, const Projection & y) const;
template void Builder::calculateRope<2, true>(Projection & z, const Projection & x, const Projection & y) const;
}  // namespace sah_kd_tree
