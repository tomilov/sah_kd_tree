#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>

#include <cassert>

void sah_kd_tree::Projection::splitPolygon(I dimension, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle,
                                           const thrust::device_vector<U> & polygonNode, U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z)
{
    // node of right part of splitted polygon (starting from polygonCount) is still node from previous layer

    polygon.min.resize(polygonCount + splittedPolygonCount);
    polygon.max.resize(polygonCount + splittedPolygonCount);

    auto polygonBboxBegin = thrust::make_zip_iterator(polygon.min.begin(), polygon.max.begin());
    auto polygonLeftBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, splittedPolygon.cbegin());
    using PolygonBboxInputType = IteratorValueType<decltype(polygonLeftBboxBegin)>;
    auto nodeBegin = thrust::make_zip_iterator(nodeSplitDimension.cbegin(), nodeSplitPos.cbegin());
    auto polygonNodeBegin = thrust::make_permutation_iterator(nodeBegin, polygonNode.cbegin());
    auto triangleBegin = thrust::make_zip_iterator(triangle.a, triangle.b, triangle.c, y.triangle.a, y.triangle.b, y.triangle.c, z.triangle.a, z.triangle.b, z.triangle.c);
    auto polygonTriangleBegin = thrust::make_permutation_iterator(triangleBegin, polygonTriangle.cbegin());
    auto polygonBegin = thrust::make_zip_iterator(polygonNodeBegin, polygonTriangleBegin);
    using PolygonType = IteratorValueType<decltype(polygonBegin)>;
    auto polygonRightBboxBegin = thrust::next(polygonBboxBegin, polygonCount);
    auto splittedPolygonBboxBegin = thrust::make_zip_iterator(polygonLeftBboxBegin, polygonRightBboxBegin);
    using SplittedPolygonBboxType = IteratorValueType<decltype(splittedPolygonBboxBegin)>;
    const auto toSplittedPolygon = [dimension] __host__ __device__(PolygonBboxInputType bbox, PolygonType polygon) -> SplittedPolygonBboxType {
        F min = thrust::get<0>(bbox), max = thrust::get<1>(bbox);
        assert(!(max < min));
        const auto & polygonNode = thrust::get<0>(polygon);
        I nodeSplitDimension = thrust::get<0>(polygonNode);
        F nodeSplitPos = thrust::get<1>(polygonNode);
        if (nodeSplitDimension == dimension) {
            assert(!(nodeSplitPos < min) && !(max < nodeSplitPos));
            return {{min, nodeSplitPos}, {nodeSplitPos, max}};
        } else if (!(min < max)) {
            return {bbox, bbox};
        }

        const auto & triangle = thrust::get<1>(polygon);
        F a, b, c;
        if (nodeSplitDimension == ((dimension + 1) % 3)) {
            a = thrust::get<3>(triangle);
            b = thrust::get<4>(triangle);
            c = thrust::get<5>(triangle);
        } else {
            assert(nodeSplitDimension == ((dimension + 2) % 3));
            a = thrust::get<6>(triangle);
            b = thrust::get<7>(triangle);
            c = thrust::get<8>(triangle);
        }

        bool aSide = (a < nodeSplitPos);
        bool bSide = (b < nodeSplitPos);
        bool cSide = (c < nodeSplitPos);

        F ax = thrust::get<0>(triangle);
        F bx = thrust::get<1>(triangle);
        F cx = thrust::get<2>(triangle);

        if (aSide == bSide) {
            assert(aSide != cSide);
            thrust::swap(aSide, cSide);
            thrust::swap(ax, cx);
            thrust::swap(a, c);
        } else if (aSide == cSide) {
            assert(aSide != bSide);
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
        F lpos = ax + (bx - ax) * (nodeSplitPos - a) / (b - a);
        assert((c < a) || (a < c));
        F rpos = ax + (cx - ax) * (nodeSplitPos - a) / (c - a);

        if (std::is_floating_point_v<F>) {
            if (max < lpos) {
                lpos = max;
            }
            if (rpos < min) {
                rpos = min;
            }

            if (rpos < lpos) {
                rpos = lpos = (lpos + rpos) / F(2);
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
    thrust::transform(polygonLeftBboxBegin, thrust::next(polygonLeftBboxBegin, splittedPolygonCount), thrust::next(polygonBegin, polygonCount), splittedPolygonBboxBegin, toSplittedPolygon);
}
