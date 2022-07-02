#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
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

void sah_kd_tree::Projection::mergeEvent(U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & polygonNode, const thrust::device_vector<U> & splittedPolygon)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto polygonBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.min.cbegin(), polygon.max.cbegin()));
    auto isPlanarPolygon = thrust::zip_function([] __host__ __device__(F min, F max) -> bool { return !(min < max); });

    auto splittedPolygonLeftBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, splittedPolygon.cbegin());
    auto splittedPolygonRightBboxBegin = thrust::next(polygonBboxBegin, polygonCount);

    auto splittedPlanarPolygonLeftCount = U(thrust::count_if(splittedPolygonLeftBboxBegin, thrust::next(splittedPolygonLeftBboxBegin, splittedPolygonCount), isPlanarPolygon));
    auto splittedPlanarPolygonRightCount = U(thrust::count_if(splittedPolygonRightBboxBegin, thrust::next(splittedPolygonRightBboxBegin, splittedPolygonCount), isPlanarPolygon));
    timer(" mergeEvent 2 * count_if");  // 0.164ms

    U splittedEventLeftCount = splittedPolygonCount * 2 - splittedPlanarPolygonLeftCount;
    U splittedEventRightCount = splittedPolygonCount * 2 - splittedPlanarPolygonRightCount;

    U splittedEventCount = splittedEventLeftCount + splittedEventRightCount;

    const auto & eventLeft = event.polygonCountLeft;
    const auto & eventRight = event.polygonCountRight;

    auto eventLeftCount = U(eventLeft.size());
    auto eventRightCount = U(eventRight.size());

    U eventStorageSize = std::exchange(eventCount, eventLeftCount + eventRightCount + splittedEventCount);
    if (eventStorageSize < eventCount) {
        eventStorageSize = eventCount;
    }

    // grant additional storage for all merge operations
    event.node.resize(eventStorageSize + eventCount);
    event.pos.resize(eventStorageSize + eventCount);
    event.kind.resize(eventStorageSize + eventCount, I(0));
    event.polygon.resize(eventStorageSize + eventCount);

    // merge l and r event
    auto eventNodeBegin = thrust::make_permutation_iterator(polygonNode.cbegin(), event.polygon.cbegin());
    auto eventValueBegin = thrust::make_zip_iterator(thrust::make_tuple(event.pos.begin(), event.kind.begin(), event.polygon.begin()));

    auto eventKeyLeftBegin = thrust::make_permutation_iterator(eventNodeBegin, eventLeft.cbegin());
    auto eventValueLeftBegin = thrust::make_permutation_iterator(eventValueBegin, eventLeft.cbegin());

    auto eventKeyRightBegin = thrust::make_permutation_iterator(eventNodeBegin, eventRight.cbegin());
    auto eventValueRightBegin = thrust::make_permutation_iterator(eventValueBegin, eventRight.cbegin());

    auto eventBothKeyBegin = thrust::next(event.node.begin(), eventStorageSize);
    auto eventBothValueBegin = thrust::next(eventValueBegin, eventStorageSize);
    thrust::merge_by_key(eventKeyLeftBegin, thrust::next(eventKeyLeftBegin, eventLeftCount), eventKeyRightBegin, thrust::next(eventKeyRightBegin, eventRightCount), eventValueLeftBegin, eventValueRightBegin, eventBothKeyBegin, eventBothValueBegin);
    timer(" mergeEvent merge_by_key");  // 12.407ms

    auto splittedEventOffset = eventStorageSize + eventLeftCount + eventRightCount;

    // calculate event kind for event of splitted polygon
    auto eventLeftKindLeftBegin = thrust::next(event.kind.begin(), splittedEventOffset);
    auto eventRightKindLeftBegin = thrust::next(eventLeftKindLeftBegin, splittedEventLeftCount);
    auto eventLeftKindRightBegin = thrust::make_reverse_iterator(eventRightKindLeftBegin);
    auto eventRightKindRightBegin = thrust::prev(eventLeftKindRightBegin, splittedEventRightCount);

    auto eventLeftKindBothBegin = thrust::make_zip_iterator(thrust::make_tuple(eventLeftKindLeftBegin, eventLeftKindRightBegin));
    auto eventRightKindBothBegin = thrust::make_zip_iterator(thrust::make_tuple(eventRightKindLeftBegin, eventRightKindRightBegin));

    auto kindBoth = thrust::make_tuple(+1, -1);
    thrust::fill_n(eventLeftKindBothBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount, kindBoth);
    thrust::fill_n(eventRightKindBothBegin, splittedPolygonCount - splittedPlanarPolygonRightCount, kindBoth);
    timer(" mergeEvent 2 * fill_n");  // 0.012ms

    // calculate left and right polygon for event of splitted polygon
    auto eventLeftPolygonLeftBegin = thrust::next(event.polygon.begin(), splittedEventOffset);
    auto eventRightPolygonLeftBegin = thrust::next(eventLeftPolygonLeftBegin, splittedEventLeftCount);
    auto eventLeftPolygonRightBegin = thrust::make_reverse_iterator(eventRightPolygonLeftBegin);
    auto eventRightPolygonRightBegin = thrust::prev(eventLeftPolygonRightBegin, splittedEventRightCount);

    auto eventLeftPolygonBothBegin = thrust::make_zip_iterator(thrust::make_tuple(eventLeftPolygonLeftBegin, eventLeftPolygonRightBegin));
    auto eventRightPolygonBothBegin = thrust::make_zip_iterator(thrust::make_tuple(eventRightPolygonLeftBegin, eventRightPolygonRightBegin));

    auto eventLeftPolygonBothOutputBegin = thrust::make_transform_output_iterator(eventLeftPolygonBothBegin, doubler<U>{});
    auto eventRightPolygonBothOutputBegin = thrust::make_transform_output_iterator(eventRightPolygonBothBegin, doubler<U>{});

    auto eventLeftPolygonPlanarBegin = thrust::next(eventLeftPolygonLeftBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount);
    auto eventRightPolygonPlanarBegin = thrust::next(eventRightPolygonLeftBegin, splittedPolygonCount - splittedPlanarPolygonRightCount);

    auto polygonLeftBegin = splittedPolygon.cbegin();
    auto polygonRightBegin = thrust::make_counting_iterator<U>(polygonCount);

    [[maybe_unused]] auto endsLeft = thrust::partition_copy(polygonLeftBegin, thrust::next(polygonLeftBegin, splittedPolygonCount), splittedPolygonLeftBboxBegin, eventLeftPolygonPlanarBegin, eventLeftPolygonBothOutputBegin, isPlanarPolygon);
    assert(thrust::next(eventLeftPolygonPlanarBegin, splittedPlanarPolygonLeftCount) == endsLeft.first);
    [[maybe_unused]] auto endsRight = thrust::partition_copy(polygonRightBegin, thrust::next(polygonRightBegin, splittedPolygonCount), splittedPolygonRightBboxBegin, eventRightPolygonPlanarBegin, eventRightPolygonBothOutputBegin, isPlanarPolygon);
    assert(thrust::next(eventRightPolygonPlanarBegin, splittedPlanarPolygonRightCount) == endsRight.first);
    timer(" mergeEvent 2 * partition_copy");  // 0.030ms

    // calculate event pos for splitted polygon
#if 1
    auto eventPolygonBegin = eventLeftPolygonLeftBegin;
    auto eventPolygonEnd = thrust::next(eventLeftPolygonLeftBegin, splittedPolygonCount);
    auto eventPosBegin = thrust::next(event.pos.begin(), splittedEventOffset);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    eventPolygonEnd = thrust::next(eventPolygonBegin, splittedPolygonCount - splittedPlanarPolygonLeftCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.max.cbegin(), eventPosBegin);
    eventPolygonEnd = thrust::next(eventPolygonBegin, splittedPolygonCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    assert(event.polygon.end() == thrust::next(eventPolygonBegin, splittedPolygonCount - splittedPlanarPolygonRightCount));
    eventPosBegin = thrust::gather(eventPolygonBegin, event.polygon.end(), polygon.max.cbegin(), eventPosBegin);
    assert(eventPosBegin == event.pos.end());
    timer(" mergeEvent 4 * gather");  // 0.005ms
#else
    auto eventPolygonBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, eventLeftPolygonLeftBegin);
    using BboxType = IteratorValueType<decltype(polygonBboxBegin)>;
    auto toEventPos = [] __host__ __device__(I eventKind, BboxType bbox) -> F { return (eventKind < 0) ? thrust::get<1>(bbox) : thrust::get<0>(bbox); };
    thrust::transform(eventLeftKindLeftBegin, event.kind.end(), eventPolygonBboxBegin, thrust::next(event.pos.begin(), splittedEventOffset), toEventPos);
    timer(" mergeEvent transform");
#endif

    // calculate event node for splitted polygon
    [[maybe_unused]] auto splittedEventNodeEnd = thrust::gather(eventLeftPolygonLeftBegin, event.polygon.end(), polygonNode.cbegin(), thrust::next(event.node.begin(), splittedEventOffset));
    assert(splittedEventNodeEnd == event.node.end());
    timer(" mergeEvent gather");  // 0.003ms

    auto eventBegin = thrust::make_zip_iterator(thrust::make_tuple(event.node.begin(), event.pos.begin(), event.kind.begin(), event.polygon.begin()));

    // sort splitted event
    auto splittedEventBegin = thrust::next(eventBegin, splittedEventOffset);
    auto splittedEventEnd = thrust::next(splittedEventBegin, splittedEventCount);
    thrust::sort(splittedEventBegin, splittedEventEnd);
    timer(" mergeEvent sort");  // 0.005ms

    // cleanup repeating planar events
    if (std::is_floating_point_v<F>) {
        auto cleanSplittedEventEnd = thrust::unique(splittedEventBegin, splittedEventEnd);
        timer(" mergeEvent unique");  // 0.203ms
        eventCount -= U(thrust::distance(cleanSplittedEventEnd, splittedEventEnd));
        splittedEventEnd = cleanSplittedEventEnd;
    }

    // merge splitted event w/ lr event
    auto eventBothBegin = thrust::next(eventBegin, eventStorageSize);
    [[maybe_unused]] auto eventEnd = thrust::merge(eventBothBegin, splittedEventBegin, splittedEventBegin, splittedEventEnd, eventBegin);
    assert(thrust::next(eventBegin, eventCount) == eventEnd);
    timer(" mergeEvent merge");  // 2.489ms

    assert(thrust::is_sorted(eventBegin, eventEnd));

    // crop
    event.node.resize(eventCount);
    event.pos.resize(eventCount);
    event.kind.resize(eventCount);
    event.polygon.resize(eventCount);
    timer(" mergeEvent crop");  // 0.010ms

    assert(thrust::equal(event.node.cbegin(), event.node.cend(), thrust::make_permutation_iterator(polygonNode.cbegin(), event.polygon.cbegin())));
}
