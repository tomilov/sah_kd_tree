#pragma once

#include <sah_kd_tree_vk/fwd.hpp>

#include <vector>

#include <cstdint>

#include <sah_kd_tree_vk/sah_kd_tree_vk_export.h>


namespace sah_kd_tree_vk
{

struct SAH_KD_TREE_VK_EXPORT Settings
{
    std::vector<std::byte> deviceUuid;
    size_t minAlignment;

    float emptinessFactor;
    float traversalCost;
    float intersectionCost;
    uint32_t maxDepth;

    void check() const;
};

}
