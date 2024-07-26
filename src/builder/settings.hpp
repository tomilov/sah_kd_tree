#pragma once

#include <sah_kd_tree_fd/fwd.hpp>

#include <vector>

#include <cstdint>

#include <sah_kd_tree_fd/sah_kd_tree_fd_export.h>


namespace sah_kd_tree_fd
{

struct SAH_KD_TREE_FD_EXPORT Settings
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
