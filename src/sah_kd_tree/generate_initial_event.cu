#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

void sah_kd_tree::Projection::generateInitialEvent()
{
    auto triangleBboxBegin = thrust::make_zip_iterator(polygon.min.cbegin(), polygon.max.cbegin());
    // using BboxType = thrust::iterator_value_t<decltype(triangleBboxBegin)>;
    const auto isPlanarEvent = thrust::make_zip_function([] __host__ __device__(F min, F max) -> bool { return !(min < max); });

    auto planarEventCount = U(thrust::count_if(triangleBboxBegin, thrust::next(triangleBboxBegin, triangle.count), isPlanarEvent));

    event.count = triangle.count - planarEventCount + triangle.count;

    event.node.resize(event.count, U(0));
    event.pos.resize(event.count);
    event.kind.resize(event.count, I(0));
    event.polygon.resize(event.count);

    auto eventKindBothBegin = thrust::make_zip_iterator(event.kind.begin(), event.kind.rbegin());
    [[maybe_unused]] auto planarEventKind = thrust::fill_n(eventKindBothBegin, triangle.count - planarEventCount, thrust::make_tuple<I, I>(+1, -1));  // right events are sequenced before left events if positions are equivalent
    // thrust::fill_n(thrust::get<0>(planarEventKind.get_iterator_tuple()), planarEventCount, I(0));

    auto triangleBegin = thrust::make_counting_iterator<U>(0);
    auto planarEventBegin = thrust::next(event.polygon.begin(), triangle.count - planarEventCount);
    auto eventPairBegin = thrust::make_zip_iterator(event.polygon.begin(), event.polygon.rbegin());
    auto solidEventBegin = thrust::make_transform_output_iterator(eventPairBegin, toPair);
    thrust::partition_copy(triangleBegin, thrust::next(triangleBegin, triangle.count), triangleBboxBegin, planarEventBegin, solidEventBegin, isPlanarEvent);

    auto eventPolygonBboxBegin = thrust::make_permutation_iterator(triangleBboxBegin, event.polygon.cbegin());
    thrust::transform(event.kind.cbegin(), event.kind.cend(), eventPolygonBboxBegin, event.pos.begin(), toEventPos);

    auto eventBegin = thrust::make_zip_iterator(event.pos.begin(), event.kind.begin(), event.polygon.begin());
    thrust::sort(eventBegin, thrust::next(eventBegin, event.count));
}
