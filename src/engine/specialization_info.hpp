#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT SpecializationInfo final : utils::NonCopyable
{
    template<
        typename SpecializationData,
        typename SpecializationMap>
    SpecializationInfo(
        std::unique_ptr<SpecializationData> && specializationDataIn,
        SpecializationMap && specializationMapIn)
    {
        specializationInfo.setData<SpecializationData>(*specializationDataIn);
        specializationData = std::move(specializationDataIn);

        specializationMap.assign_range(std::forward<SpecializationMap>(specializationMapIn));
        specializationInfo.setMapEntries(specializationMap);
    }

    [[nodiscard]] const vk::SpecializationInfo & getSpecializationInfo() const &
    {
        return specializationInfo;
    }

    [[nodiscard]] operator const vk::SpecializationInfo &() const &  // NOLINT: google-explicit-constructor
    {
        return getSpecializationInfo();
    }

private:
    std::shared_ptr<void> specializationData;
    std::vector<vk::SpecializationMapEntry> specializationMap;

    vk::SpecializationInfo specializationInfo;
};

using SpecializationInfos = std::unordered_map<vk::ShaderStageFlagBits, SpecializationInfo>;

}  // namespace engine
