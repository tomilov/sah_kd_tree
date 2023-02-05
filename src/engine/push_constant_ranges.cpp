#include <engine/push_constant_ranges.hpp>
#include <utils/assert.hpp>

#include <format/vulkan.hpp>

#include <algorithm>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

namespace engine
{

namespace
{

struct PushConstantRangeLess
{
    bool operator()(const vk::PushConstantRange & lhs, const vk::PushConstantRange & rhs) const noexcept
    {
        if (lhs.stageFlags == rhs.stageFlags) {
            return lhs.offset + lhs.size <= rhs.offset;
        }
        return lhs.stageFlags < rhs.stageFlags;
    }
};

struct PushConstantRangeBoundary
{
    uint32_t offset;
    bool isLeft;
    vk::ShaderStageFlags stageFlags;

    bool operator<(const PushConstantRangeBoundary & rhs) const noexcept
    {
        if (offset == rhs.offset) {
            if (isLeft == rhs.isLeft) {
                return false;
            }
            return !isLeft;
        }
        return offset < rhs.offset;
    }
};

}  // namespace

std::vector<vk::PushConstantRange> getDisjointPushConstantRanges(const std::vector<vk::PushConstantRange> & pushConstantRanges)
{
    if (std::size(pushConstantRanges) < 2) {
        return pushConstantRanges;
    }
    {
        std::set<vk::PushConstantRange, PushConstantRangeLess> singleLayerCover;
        for (const auto & pushConstantRange : pushConstantRanges) {
            auto [it, inserted] = singleLayerCover.insert(pushConstantRange);
            INVARIANT(inserted, "Push constant ranges {} and {} has common subrange", pushConstantRange, *it);
        }
    }
    std::vector<PushConstantRangeBoundary> pushConstantRangeBoundaries;
    pushConstantRangeBoundaries.reserve(std::size(pushConstantRanges));
    for (const auto & pushConstantRange : pushConstantRanges) {
        INVARIANT(pushConstantRange.size != 0, "");
        pushConstantRangeBoundaries.push_back({pushConstantRange.offset, true, pushConstantRange.stageFlags});
        pushConstantRangeBoundaries.push_back({pushConstantRange.offset + pushConstantRange.size, false, pushConstantRange.stageFlags});
    }
    std::sort(std::begin(pushConstantRangeBoundaries), std::end(pushConstantRangeBoundaries));
    {
        auto l = std::begin(pushConstantRangeBoundaries);
        auto r = std::next(l);
        do {
            do {
                if (*l < *r) {
                    *++l = *r++;
                    break;
                }
                l->stageFlags |= r->stageFlags;
            } while (++r != std::end(pushConstantRangeBoundaries));
        } while (r != std::end(pushConstantRangeBoundaries));
        pushConstantRangeBoundaries.erase(std::next(l), r);
    }
    std::vector<vk::PushConstantRange> disjointPushConstantRanges;
    disjointPushConstantRanges.reserve(std::size(pushConstantRangeBoundaries));
    for (const auto & pushConstantRangeBoundary : pushConstantRangeBoundaries) {
        if (std::empty(disjointPushConstantRanges)) {
            ASSERT(pushConstantRangeBoundary.isLeft);
            disjointPushConstantRanges.push_back({pushConstantRangeBoundary.stageFlags, pushConstantRangeBoundary.offset, 0});
        } else {
            auto & disjointPushConstantRange = disjointPushConstantRanges.back();
            if (pushConstantRangeBoundary.isLeft) {
                ASSERT(&pushConstantRangeBoundary != &pushConstantRangeBoundaries.back());
                ASSERT(disjointPushConstantRange.offset <= pushConstantRangeBoundary.offset);
                if (disjointPushConstantRange.size == 0) {
                    if ((disjointPushConstantRange.stageFlags & pushConstantRangeBoundary.stageFlags) != pushConstantRangeBoundary.stageFlags) {
                        disjointPushConstantRange.size = pushConstantRangeBoundary.offset - disjointPushConstantRange.offset;
                        disjointPushConstantRanges.push_back({disjointPushConstantRange.stageFlags | pushConstantRangeBoundary.stageFlags, pushConstantRangeBoundary.offset, 0});
                    }
                } else {
                    if ((disjointPushConstantRange.stageFlags == pushConstantRangeBoundary.stageFlags) && (disjointPushConstantRange.size + disjointPushConstantRange.offset == pushConstantRangeBoundary.offset)) {
                        ASSERT(disjointPushConstantRange.size != 0);
                        disjointPushConstantRange.size = 0;
                    } else {
                        disjointPushConstantRanges.push_back({pushConstantRangeBoundary.stageFlags, pushConstantRangeBoundary.offset, 0});
                    }
                }
            } else {
                ASSERT(disjointPushConstantRange.offset < pushConstantRangeBoundary.offset);
                disjointPushConstantRange.size = pushConstantRangeBoundary.offset - disjointPushConstantRange.offset;
                if (disjointPushConstantRange.stageFlags != pushConstantRangeBoundary.stageFlags) {
                    ASSERT(&pushConstantRangeBoundary != &pushConstantRangeBoundaries.back());
                    ASSERT((disjointPushConstantRange.stageFlags & pushConstantRangeBoundary.stageFlags) == pushConstantRangeBoundary.stageFlags);
                    disjointPushConstantRanges.push_back({disjointPushConstantRange.stageFlags & ~pushConstantRangeBoundary.stageFlags, pushConstantRangeBoundary.offset, 0});
                }
            }
        }
    }
    return disjointPushConstantRanges;
}

}  // namespace engine
