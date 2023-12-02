#include <engine/push_constant_ranges.hpp>
#include <engine/utils.hpp>
#include <format/vulkan.hpp>

#include <bit>
#include <iterator>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace engine
{

namespace
{

struct PushConstantRangeLess
{
    bool operator()(const vk::PushConstantRange & lhs, const vk::PushConstantRange & rhs) const noexcept
    {
        return lhs.offset + lhs.size <= rhs.offset;
    }
};

void widenPushConstantRange(vk::PushConstantRange & lhs, const vk::PushConstantRange & rhs)
{
    if (lhs.offset + lhs.size < rhs.offset + rhs.size) {
        lhs.size = rhs.offset + rhs.size - lhs.offset;
    }
    if (lhs.offset > rhs.offset) {
        lhs.size += lhs.offset - rhs.offset;
        lhs.offset = rhs.offset;
    }
}

}  // namespace

std::vector<vk::PushConstantRange> mergePushConstantRanges(std::span<const vk::PushConstantRange> pushConstantRanges)
{
    using MaskType = typename vk::ShaderStageFlags::MaskType;
    vk::PushConstantRange stagePushContantRanges[std::numeric_limits<MaskType>::digits];
    for (const auto & pushConstantRange : pushConstantRanges) {
        for (vk::ShaderStageFlagBits stageFlagBit : FlagBits{pushConstantRange.stageFlags}) {
            auto & stagePushContantRange = stagePushContantRanges[std::countr_zero(static_cast<MaskType>(stageFlagBit))];
            if (stagePushContantRange.stageFlags) {
                ASSERT(stagePushContantRange.stageFlags == stageFlagBit);
                widenPushConstantRange(stagePushContantRange, pushConstantRange);
            } else {
                stagePushContantRange = {
                    .stageFlags = stageFlagBit,
                    .offset = pushConstantRange.offset,
                    .size = pushConstantRange.size,
                };
            }
        }
    }
    std::set<vk::PushConstantRange, PushConstantRangeLess> mergedPushConstantRanges;
    for (auto & stagePushContantRange : stagePushContantRanges) {
        if (!stagePushContantRange.stageFlags) {
            continue;
        }
        const auto [l, r] = mergedPushConstantRanges.equal_range(stagePushContantRange);
        for (auto it = l; it != r; ++it) {
            ASSERT(!(stagePushContantRange.stageFlags & it->stageFlags));
            stagePushContantRange.stageFlags |= it->stageFlags;
            widenPushConstantRange(stagePushContantRange, *it);
        }
        mergedPushConstantRanges.insert(mergedPushConstantRanges.erase(l, r), stagePushContantRange);
    }
    return {std::cbegin(mergedPushConstantRanges), std::cend(mergedPushConstantRanges)};
}

}  // namespace engine
