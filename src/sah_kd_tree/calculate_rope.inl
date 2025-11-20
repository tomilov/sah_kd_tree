#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/transform.h>

#include <cassert>

namespace sah_kd_tree
{
template<typename Traits>
template<typename Traits::I dimension, bool forth>
void Builder<Traits>::calculateRope(Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z) const
{
    auto & nodeRope = forth ? x.node.rightRope : x.node.leftRope;
    nodeRope.resize(node.count);

    auto yMins = thrust::raw_pointer_cast(y.node.min.data());
    auto yMaxs = thrust::raw_pointer_cast(y.node.max.data());
    auto zMins = thrust::raw_pointer_cast(z.node.min.data());
    auto zMaxs = thrust::raw_pointer_cast(z.node.max.data());
    auto parents = thrust::raw_pointer_cast(node.parent.data());
    auto leftChildren = thrust::raw_pointer_cast(node.leftChild.data());
    auto rightChildren = thrust::raw_pointer_cast(node.rightChild.data());
    auto splitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto splitPositions = thrust::raw_pointer_cast(node.splitPos.data());
    const auto getRightRope = [yMins, yMaxs, zMins, zMaxs, parents, leftChildren, rightChildren, splitDimensions, splitPositions] __host__ __device__(U node) -> U
    {
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

}  // namespace sah_kd_tree
