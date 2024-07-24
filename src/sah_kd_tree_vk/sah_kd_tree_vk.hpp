#pragma once

#include <sah_kd_tree_vk/fwd.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <cstdint>

#include <cuda_runtime.h>

#include <sah_kd_tree_vk/sah_kd_tree_vk_export.h>

namespace sah_kd_tree_vk
{

class SAH_KD_TREE_VK_EXPORT Tree : utils::NonCopyable
{
public:
    struct Settings
    {
        float emptinessFactor;
        float traversalCost;
        float intersectionCost;
        uint32_t maxDepth;

        void check() const;
    };

    explicit Tree(const scene_data::SceneData & sceneData, const Settings & settings);
    ~Tree();

private:
    struct Impl;

    static constexpr size_t kSize = 1;
    static constexpr size_t kAlignment = 1;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace sah_kd_tree_vk
