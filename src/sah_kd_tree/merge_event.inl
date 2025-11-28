#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>

#include <type_traits>
#include <utility>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Projection<Traits>::mergeEvent(U polygonCount, U splittedPolygonCount, const Vector<U> & polygonNode, const Vector<U> & splittedPolygon)
{
    auto polygonBboxBegin = thrust::make_zip_iterator(polygon.min.cbegin(), polygon.max.cbegin());
    const auto isPlanarPolygon = thrust::make_zip_function([] __host__ __device__(F min, F max) -> bool { return !(min < max); });

    auto splittedPolygonLeftBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, splittedPolygon.cbegin());
    auto splittedPolygonRightBboxBegin = cuda::std::next(polygonBboxBegin, polygonCount);

    auto splittedPlanarPolygonLeftCount = safeConvert<U>(thrust::count_if(splittedPolygonLeftBboxBegin, cuda::std::next(splittedPolygonLeftBboxBegin, splittedPolygonCount), isPlanarPolygon));
    auto splittedPlanarPolygonRightCount = safeConvert<U>(thrust::count_if(splittedPolygonRightBboxBegin, cuda::std::next(splittedPolygonRightBboxBegin, splittedPolygonCount), isPlanarPolygon));

    U splittedEventLeftCount = splittedPolygonCount * 2 - splittedPlanarPolygonLeftCount;
    U splittedEventRightCount = splittedPolygonCount * 2 - splittedPlanarPolygonRightCount;

    U splittedEventCount = splittedEventLeftCount + splittedEventRightCount;

    const auto & eventLeft = event.polygonCountLeft;
    const auto & eventRight = event.polygonCountRight;

    auto eventLeftCount = safeConvert<U>(eventLeft.size());
    auto eventRightCount = safeConvert<U>(eventRight.size());

    U eventStorageSize = std::exchange(event.count, eventLeftCount + eventRightCount + splittedEventCount);
    if (eventStorageSize < event.count) {
        eventStorageSize = event.count;
    }

    // grant additional storage for all merge operations
    event.node.resize(eventStorageSize + event.count);
    event.pos.resize(eventStorageSize + event.count);
    event.kind.resize(eventStorageSize + event.count, static_cast<I>(0));
    event.polygon.resize(eventStorageSize + event.count);

    // merge l and r event
    auto eventNodeBegin = thrust::make_permutation_iterator(polygonNode.cbegin(), event.polygon.cbegin());
    auto eventValueBegin = thrust::make_zip_iterator(event.pos.begin(), event.kind.begin(), event.polygon.begin());

    auto eventKeyLeftBegin = thrust::make_permutation_iterator(eventNodeBegin, eventLeft.cbegin());
    auto eventValueLeftBegin = thrust::make_permutation_iterator(eventValueBegin, eventLeft.cbegin());

    auto eventKeyRightBegin = thrust::make_permutation_iterator(eventNodeBegin, eventRight.cbegin());
    auto eventValueRightBegin = thrust::make_permutation_iterator(eventValueBegin, eventRight.cbegin());

    auto eventBothKeyBegin = cuda::std::next(event.node.begin(), eventStorageSize);
    auto eventBothValueBegin = cuda::std::next(eventValueBegin, eventStorageSize);
    thrust::merge_by_key(eventKeyLeftBegin, cuda::std::next(eventKeyLeftBegin, eventLeftCount), eventKeyRightBegin, cuda::std::next(eventKeyRightBegin, eventRightCount), eventValueLeftBegin, eventValueRightBegin, eventBothKeyBegin,
                         eventBothValueBegin);

    auto splittedEventOffset = eventStorageSize + eventLeftCount + eventRightCount;

    // calculate event kind for event of splitted polygon
    auto eventLeftKindLeftBegin = cuda::std::next(event.kind.begin(), splittedEventOffset);
    auto eventRightKindLeftBegin = cuda::std::next(eventLeftKindLeftBegin, splittedEventLeftCount);
    auto eventLeftKindRightBegin = thrust::make_reverse_iterator(eventRightKindLeftBegin);
    auto eventRightKindRightBegin = cuda::std::prev(eventLeftKindRightBegin, splittedEventRightCount);

    auto eventLeftKindBothBegin = thrust::make_zip_iterator(eventLeftKindLeftBegin, eventLeftKindRightBegin);
    auto eventRightKindBothBegin = thrust::make_zip_iterator(eventRightKindLeftBegin, eventRightKindRightBegin);

    auto kindBoth = thrust::make_tuple<I, I>(+1, -1);
    thrust::fill_n(eventLeftKindBothBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount, kindBoth);
    thrust::fill_n(eventRightKindBothBegin, splittedPolygonCount - splittedPlanarPolygonRightCount, kindBoth);

    // calculate left and right polygon for event of splitted polygon
    auto eventLeftPolygonLeftBegin = cuda::std::next(event.polygon.begin(), splittedEventOffset);
    auto eventRightPolygonLeftBegin = cuda::std::next(eventLeftPolygonLeftBegin, splittedEventLeftCount);
    auto eventLeftPolygonRightBegin = thrust::make_reverse_iterator(eventRightPolygonLeftBegin);
    auto eventRightPolygonRightBegin = cuda::std::prev(eventLeftPolygonRightBegin, splittedEventRightCount);

    auto eventLeftPolygonBothBegin = thrust::make_zip_iterator(eventLeftPolygonLeftBegin, eventLeftPolygonRightBegin);
    auto eventRightPolygonBothBegin = thrust::make_zip_iterator(eventRightPolygonLeftBegin, eventRightPolygonRightBegin);

    auto eventLeftPolygonBothOutputBegin = thrust::make_transform_output_iterator(eventLeftPolygonBothBegin, toPair);
    auto eventRightPolygonBothOutputBegin = thrust::make_transform_output_iterator(eventRightPolygonBothBegin, toPair);

    auto eventLeftPolygonPlanarBegin = cuda::std::next(eventLeftPolygonLeftBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount);
    auto eventRightPolygonPlanarBegin = cuda::std::next(eventRightPolygonLeftBegin, splittedPolygonCount - splittedPlanarPolygonRightCount);

    auto polygonLeftBegin = splittedPolygon.cbegin();
    auto polygonRightBegin = thrust::make_counting_iterator<U>(polygonCount);

    [[maybe_unused]] auto endsLeft = thrust::partition_copy(polygonLeftBegin, cuda::std::next(polygonLeftBegin, splittedPolygonCount), splittedPolygonLeftBboxBegin, eventLeftPolygonPlanarBegin, eventLeftPolygonBothOutputBegin, isPlanarPolygon);
    assert(cuda::std::next(eventLeftPolygonPlanarBegin, splittedPlanarPolygonLeftCount) == endsLeft.first);
    [[maybe_unused]] auto endsRight = thrust::partition_copy(polygonRightBegin, cuda::std::next(polygonRightBegin, splittedPolygonCount), splittedPolygonRightBboxBegin, eventRightPolygonPlanarBegin, eventRightPolygonBothOutputBegin, isPlanarPolygon);
    assert(cuda::std::next(eventRightPolygonPlanarBegin, splittedPlanarPolygonRightCount) == endsRight.first);

    // calculate event pos for splitted polygon
#if 0
    auto eventPolygonBegin = eventLeftPolygonLeftBegin;
    auto eventPolygonEnd = cuda::std::next(eventLeftPolygonLeftBegin, splittedPolygonCount);
    auto eventPosBegin = cuda::std::next(event.pos.begin(), splittedEventOffset);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    eventPolygonEnd = cuda::std::next(eventPolygonBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.max.cbegin(), eventPosBegin);
    eventPolygonEnd = cuda::std::next(eventPolygonBegin, splittedPolygonCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    assert(event.polygon.end() == cuda::std::next(eventPolygonBegin, splittedPolygonCount - splittedPlanarPolygonRightCount));
    eventPosBegin = thrust::gather(eventPolygonBegin, event.polygon.end(), polygon.max.cbegin(), eventPosBegin);
    assert(eventPosBegin == event.pos.end());
#else
    auto eventPolygonBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, eventLeftPolygonLeftBegin);
    // using BboxType = cuda::std::iter_value_t<decltype(polygonBboxBegin)>;
    thrust::transform(eventLeftKindLeftBegin, event.kind.end(), eventPolygonBboxBegin, cuda::std::next(event.pos.begin(), splittedEventOffset), toEventPos);
#endif

    // calculate event node for splitted polygon
    [[maybe_unused]] auto splittedEventNodeEnd = thrust::gather(eventLeftPolygonLeftBegin, event.polygon.end(), polygonNode.cbegin(), cuda::std::next(event.node.begin(), splittedEventOffset));
    assert(splittedEventNodeEnd == event.node.end());

    auto eventBegin = thrust::make_zip_iterator(event.node.begin(), event.pos.begin(), event.kind.begin(), event.polygon.begin());

    // sort splitted event
    auto splittedEventBegin = cuda::std::next(eventBegin, splittedEventOffset);
    auto splittedEventEnd = cuda::std::next(splittedEventBegin, splittedEventCount);
    thrust::sort(splittedEventBegin, splittedEventEnd);  // max sqrt(N) * log(sqrt(N)) for "reasonable" scenes

    // cleanup repeating planar events
    if (std::is_floating_point_v<F>) {
        auto cleanSplittedEventEnd = thrust::unique(splittedEventBegin, splittedEventEnd);
        event.count -= safeConvert<U>(cuda::std::distance(cleanSplittedEventEnd, splittedEventEnd));
        splittedEventEnd = cleanSplittedEventEnd;
    }

    // merge splitted event w/ lr event
    auto eventBothBegin = cuda::std::next(eventBegin, eventStorageSize);
    [[maybe_unused]] auto eventEnd = thrust::merge(eventBothBegin, splittedEventBegin, splittedEventBegin, splittedEventEnd, eventBegin);
    assert(cuda::std::next(eventBegin, event.count) == eventEnd);

    assert(thrust::is_sorted(eventBegin, eventEnd));

    // crop
    event.node.resize(event.count);
    event.pos.resize(event.count);
    event.kind.resize(event.count);
    event.polygon.resize(event.count);

    assert(thrust::equal(event.node.cbegin(), event.node.cend(), thrust::make_permutation_iterator(polygonNode.cbegin(), event.polygon.cbegin())));
}
