#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

void sah_kd_tree::Projection::generateInitialEvent(U triangleCount)
{
    auto triangleBboxBegin = thrust::make_zip_iterator(polygon.min.cbegin(), polygon.max.cbegin());
    using BboxType = IteratorValueType<decltype(triangleBboxBegin)>;
    const auto isPlanarEvent = thrust::zip_function([] __host__ __device__(F min, F max) -> bool { return !(min < max); });

    auto planarEventCount = U(thrust::count_if(triangleBboxBegin, thrust::next(triangleBboxBegin, triangleCount), isPlanarEvent));

    auto eventCount = triangleCount - planarEventCount + triangleCount;

    event.node.resize(eventCount, U(0));
    event.pos.resize(eventCount);
    event.kind.resize(eventCount, I(0));
    event.polygon.resize(eventCount);

    auto eventKindBothBegin = thrust::make_zip_iterator(event.kind.begin(), event.kind.rbegin());
    [[maybe_unused]] auto planarEventKind = thrust::fill_n(eventKindBothBegin, triangleCount - planarEventCount, thrust::make_tuple<I, I>(+1, -1));  // right events are sequenced before left events if positions are equivalent
    // thrust::fill_n(thrust::get<0>(planarEventKind.get_iterator_tuple()), planarEventCount, I(0));

    auto triangleBegin = thrust::make_counting_iterator<U>(0);
    auto planarEventBegin = thrust::next(event.polygon.begin(), triangleCount - planarEventCount);
    auto eventPairBegin = thrust::make_zip_iterator(event.polygon.begin(), event.polygon.rbegin());
    auto solidEventBegin = thrust::make_transform_output_iterator(eventPairBegin, doubler<U>{});
    thrust::partition_copy(triangleBegin, thrust::next(triangleBegin, triangleCount), triangleBboxBegin, planarEventBegin, solidEventBegin, isPlanarEvent);

    auto eventPolygonBboxBegin = thrust::make_permutation_iterator(triangleBboxBegin, event.polygon.cbegin());
    const auto toEventPos = [] __host__ __device__(I eventKind, BboxType bbox) -> F { return (eventKind < 0) ? thrust::get<1>(bbox) : thrust::get<0>(bbox); };
    thrust::transform(event.kind.cbegin(), event.kind.cend(), eventPolygonBboxBegin, event.pos.begin(), toEventPos);

    auto eventBegin = thrust::make_zip_iterator(event.pos.begin(), event.kind.begin(), event.polygon.begin());
    thrust::sort(eventBegin, thrust::next(eventBegin, eventCount));
}
