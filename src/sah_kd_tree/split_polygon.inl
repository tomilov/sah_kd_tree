#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/swap.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>

#include <cassert>

namespace sah_kd_tree
{
template<typename Traits>
template<typename Traits::I dimension>
void sah_kd_tree::Builder<Traits>::splitPolygon(Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z) const
{
    // node of right part of splitted polygon (starting from polygon.count) is still node from previous layer

    x.polygon.min.resize(polygon.count + polygon.splittedCount);
    x.polygon.max.resize(polygon.count + polygon.splittedCount);

    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto nodeSplitPositions = thrust::raw_pointer_cast(node.splitPos.data());
    auto polygonNodes = thrust::raw_pointer_cast(polygon.node.data());

    auto polygonTriangles = thrust::raw_pointer_cast(polygon.triangle.data());

    auto AX = thrust::raw_pointer_cast(x.triangle.a);
    auto BX = thrust::raw_pointer_cast(x.triangle.b);
    auto CX = thrust::raw_pointer_cast(x.triangle.c);

    auto AY = thrust::raw_pointer_cast(y.triangle.a);
    auto BY = thrust::raw_pointer_cast(y.triangle.b);
    auto CY = thrust::raw_pointer_cast(y.triangle.c);

    auto AZ = thrust::raw_pointer_cast(z.triangle.a);
    auto BZ = thrust::raw_pointer_cast(z.triangle.b);
    auto CZ = thrust::raw_pointer_cast(z.triangle.c);

    auto polygonBboxBegin = thrust::make_zip_iterator(x.polygon.min.begin(), x.polygon.max.begin());
    using PolygonBboxInputType = cuda::std::iter_value_t<decltype(polygonBboxBegin)>;
    auto polygonLeftBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, splittedPolygon.cbegin());
    auto polygonRightBboxBegin = cuda::std::next(polygonBboxBegin, polygon.count);
    auto splittedPolygonBboxBegin = thrust::make_zip_iterator(polygonLeftBboxBegin, polygonRightBboxBegin);
    using SplittedPolygonBboxType = cuda::std::iter_value_t<decltype(splittedPolygonBboxBegin)>;

    const auto toSplittedPolygon = [polygonNodes, nodeSplitDimensions, nodeSplitPositions, polygonTriangles, AX, BX, CX, AY, BY, CY, AZ, BZ, CZ] __host__ __device__(PolygonBboxInputType bbox, U polygon) -> SplittedPolygonBboxType
    {
        F min = thrust::get<0>(bbox), max = thrust::get<1>(bbox);
        assert(!(max < min));
        U polygonNode = polygonNodes[polygon];
        I polygonSplitDimension = nodeSplitDimensions[polygonNode];
        F polygonSplitPos = nodeSplitPositions[polygonNode];
        if (polygonSplitDimension == dimension) {
            assert(!(polygonSplitPos < min) && !(max < polygonSplitPos));
            return {{min, polygonSplitPos}, {polygonSplitPos, max}};
        } else if (!(min < max)) {
            return {bbox, bbox};
        }

        U triangle = polygonTriangles[polygon];
        F a, b, c;
        if (polygonSplitDimension == ((dimension + 1) % 3)) {
            a = AY[triangle];
            b = BY[triangle];
            c = CY[triangle];
        } else {
            assert(polygonSplitDimension == ((dimension + 2) % 3));
            a = AZ[triangle];
            b = BZ[triangle];
            c = CZ[triangle];
        }

        bool aSide = (a < polygonSplitPos);
        bool bSide = (b < polygonSplitPos);
        bool cSide = (c < polygonSplitPos);

        F ax = AX[triangle];
        F bx = BX[triangle];
        F cx = CX[triangle];

        if (aSide == bSide) {
            assert(aSide != cSide);
            thrust::swap(aSide, cSide);
            thrust::swap(ax, cx);
            thrust::swap(a, c);
        } else if (aSide == cSide) {
            thrust::swap(aSide, bSide);
            thrust::swap(ax, bx);
            thrust::swap(a, b);
        }
        assert(bSide == cSide);

        if (aSide != ((bx - ax) * (c - a) < (b - a) * (cx - ax))) {
            thrust::swap(bx, cx);
            thrust::swap(b, c);
        }

        assert((b < a) || (a < b));
        F lpos = ax + (bx - ax) * (polygonSplitPos - a) / (b - a);
        assert((c < a) || (a < c));
        F rpos = ax + (cx - ax) * (polygonSplitPos - a) / (c - a);

        if (std::is_floating_point_v<F>) {
            if (max < lpos) {
                lpos = max;
            }
            if (rpos < min) {
                rpos = min;
            }

            if (rpos < lpos) {
                rpos = lpos = (lpos + rpos) / static_cast<F>(2);
            }
        }
        assert(!(rpos < lpos));

        F lmin, rmin;
        if (min < lpos) {
            if ((ax < bx) == (a < b)) {
                lmin = min;
                rmin = lpos;
            } else {
                lmin = lpos;
                rmin = min;
            }
        } else {
            lmin = min;
            rmin = min;
        }
        F lmax, rmax;
        if (rpos < max) {
            if ((ax < cx) == (a < c)) {
                lmax = rpos;
                rmax = max;
            } else {
                lmax = max;
                rmax = rpos;
            }
        } else {
            lmax = max;
            rmax = max;
        }
        assert(!(lmax < lmin));
        assert(!(rmax < rmin));
        return {{lmin, lmax}, {rmin, rmax}};
    };
    auto polygonBegin = thrust::make_counting_iterator<U>(polygon.count);
    thrust::transform(polygonLeftBboxBegin, cuda::std::next(polygonLeftBboxBegin, polygon.splittedCount), polygonBegin, splittedPolygonBboxBegin, toSplittedPolygon);
}
}  // namespace sah_kd_tree
