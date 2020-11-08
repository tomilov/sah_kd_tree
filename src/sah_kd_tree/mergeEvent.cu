#include "utility.cuh"

#include <sah_kd_tree/projection.hpp>

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

void SahKdTree::Projection::mergeEvent(U polygonCount, const thrust::device_vector<U> & polygonNode, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon)
{
    Timer timer;
    auto eventCount = U(event.kind.size());

    auto polygonBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.min.cbegin(), polygon.max.cbegin()));
    auto isPlanarPolygon = thrust::zip_function([] __host__ __device__(F min, F max) -> bool { return !(min < max); });

    auto polygonLeftBboxBegin = thrust::make_permutation_iterator(polygonBboxBegin, splittedPolygon.cbegin());
    auto polygonRightBboxBegin = thrust::next(polygonBboxBegin, polygonCount);

    auto planarPolygonLeftCount = U(thrust::count_if(polygonLeftBboxBegin, thrust::next(polygonLeftBboxBegin, splittedPolygonCount), isPlanarPolygon));
    auto planarPolygonRightCount = U(thrust::count_if(polygonRightBboxBegin, thrust::next(polygonRightBboxBegin, splittedPolygonCount), isPlanarPolygon));
    timer(" mergeEvent 2 * count_if");  // 0.164ms

    U splittedEventLeftCount = splittedPolygonCount + splittedPolygonCount - planarPolygonLeftCount;
    U splittedEventRightCount = splittedPolygonCount + splittedPolygonCount - planarPolygonRightCount;

    U splittedEventCount = splittedEventLeftCount + splittedEventRightCount;

    auto countLeft = U(event.countLeft.size());
    auto countRight = U(event.countRight.size());

    U eventStorageSize = std::exchange(eventCount, countLeft + countRight + splittedEventCount);
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

    auto eventKeyLeftBegin = thrust::make_permutation_iterator(eventNodeBegin, event.countLeft.cbegin());
    auto eventValueLeftBegin = thrust::make_permutation_iterator(eventValueBegin, event.countLeft.cbegin());

    auto eventKeyRightBegin = thrust::make_permutation_iterator(eventNodeBegin, event.countRight.cbegin());
    auto eventValueRightBegin = thrust::make_permutation_iterator(eventValueBegin, event.countRight.cbegin());

    auto eventLeftRightKeyBegin = thrust::next(event.node.begin(), eventStorageSize);
    auto eventLeftRightValueBegin = thrust::next(eventValueBegin, eventStorageSize);
    thrust::merge_by_key(eventKeyLeftBegin, thrust::next(eventKeyLeftBegin, countLeft), eventKeyRightBegin, thrust::next(eventKeyRightBegin, countRight), eventValueLeftBegin, eventValueRightBegin, eventLeftRightKeyBegin, eventLeftRightValueBegin);
    timer(" mergeEvent merge_by_key");  // 12.407ms

    auto splittedEventOffset = eventStorageSize + countLeft + countRight;

    // calculate event kind for event of splitted polygon
    auto eventLeftKindLeftBegin = thrust::next(event.kind.begin(), splittedEventOffset);
    auto eventRightKindLeftBegin = thrust::next(eventLeftKindLeftBegin, splittedEventLeftCount);
    auto eventLeftKindRightBegin = thrust::make_reverse_iterator(eventRightKindLeftBegin);
    auto eventRightKindRightBegin = thrust::prev(eventLeftKindRightBegin, splittedEventRightCount);

    auto eventLeftKindLeftRightBegin = thrust::make_zip_iterator(thrust::make_tuple(eventLeftKindLeftBegin, eventLeftKindRightBegin));
    auto eventRightKindLeftRightBegin = thrust::make_zip_iterator(thrust::make_tuple(eventRightKindLeftBegin, eventRightKindRightBegin));

    auto kindLeftRight = thrust::make_tuple(+1, -1);
    thrust::fill_n(eventLeftKindLeftRightBegin, splittedPolygonCount - planarPolygonLeftCount, kindLeftRight);
    thrust::fill_n(eventRightKindLeftRightBegin, splittedPolygonCount - planarPolygonRightCount, kindLeftRight);
    timer(" mergeEvent 2 * fill_n");  // 0.012ms

    // calculate l and r polygon for event of splitted polygon
    auto eventLeftPolygonLeftBegin = thrust::next(event.polygon.begin(), splittedEventOffset);
    auto eventRightPolygonLeftBegin = thrust::next(eventLeftPolygonLeftBegin, splittedEventLeftCount);
    auto eventLeftPolygonRightBegin = thrust::make_reverse_iterator(eventRightPolygonLeftBegin);
    auto eventRightPolygonRightBegin = thrust::prev(eventLeftPolygonRightBegin, splittedEventRightCount);

    auto eventLeftPolygonLeftRightBegin = thrust::make_zip_iterator(thrust::make_tuple(eventLeftPolygonLeftBegin, eventLeftPolygonRightBegin));
    auto eventRightPolygonLeftRightBegin = thrust::make_zip_iterator(thrust::make_tuple(eventRightPolygonLeftBegin, eventRightPolygonRightBegin));

    auto eventLeftPolygonLeftRightOutputBegin = thrust::make_transform_output_iterator(eventLeftPolygonLeftRightBegin, doubler<U>{});
    auto eventRightPolygonLeftRightOutputBegin = thrust::make_transform_output_iterator(eventRightPolygonLeftRightBegin, doubler<U>{});

    auto eventLeftPolygonPlanarBegin = thrust::next(eventLeftPolygonLeftBegin, splittedPolygonCount - planarPolygonLeftCount);
    auto eventRightPolygonPlanarBegin = thrust::next(eventRightPolygonLeftBegin, splittedPolygonCount - planarPolygonRightCount);

    auto polygonLeftBegin = splittedPolygon.cbegin();
    auto polygonRightBegin = thrust::make_counting_iterator<U>(polygonCount);

    [[maybe_unused]] auto endsLeft = thrust::partition_copy(polygonLeftBegin, thrust::next(polygonLeftBegin, splittedPolygonCount), polygonLeftBboxBegin, eventLeftPolygonPlanarBegin, eventLeftPolygonLeftRightOutputBegin, isPlanarPolygon);
    assert(thrust::next(eventLeftPolygonPlanarBegin, planarPolygonLeftCount) == endsLeft.first);
    [[maybe_unused]] auto endsRight = thrust::partition_copy(polygonRightBegin, thrust::next(polygonRightBegin, splittedPolygonCount), polygonRightBboxBegin, eventRightPolygonPlanarBegin, eventRightPolygonLeftRightOutputBegin, isPlanarPolygon);
    assert(thrust::next(eventRightPolygonPlanarBegin, planarPolygonRightCount) == endsRight.first);
    timer(" mergeEvent 2 * partition_copy");  // 0.030ms

    // calculate event pos
#if 1
    auto eventPolygonBegin = eventLeftPolygonLeftBegin;
    auto eventPolygonEnd = thrust::next(eventLeftPolygonLeftBegin, splittedPolygonCount);
    auto eventPosBegin = thrust::next(event.pos.begin(), splittedEventOffset);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    eventPolygonEnd = thrust::next(eventPolygonBegin, splittedPolygonCount - planarPolygonLeftCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.max.cbegin(), eventPosBegin);
    eventPolygonEnd = thrust::next(eventPolygonBegin, splittedPolygonCount);
    eventPosBegin = thrust::gather(std::exchange(eventPolygonBegin, eventPolygonEnd), eventPolygonEnd, polygon.min.cbegin(), eventPosBegin);
    assert(event.polygon.end() == thrust::next(eventPolygonBegin, splittedPolygonCount - planarPolygonRightCount));
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

    // calculate event node
    thrust::gather(eventLeftPolygonLeftBegin, event.polygon.end(), polygonNode.cbegin(), thrust::next(event.node.begin(), splittedEventOffset));
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
    auto eventLeftRightBegin = thrust::next(eventBegin, eventStorageSize);
    [[maybe_unused]] auto eventEnd = thrust::merge(eventLeftRightBegin, splittedEventBegin, splittedEventBegin, splittedEventEnd, eventBegin);
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
