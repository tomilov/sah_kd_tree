#pragma once

#include <builder/fwd.hpp>

#include <array>

#include <cstdint>
#include <cstddef>

#include <builder/builder_export.h>


namespace builder
{

struct BUILDER_EXPORT Settings
{
    std::array<std::byte, 16> deviceUuid;
    size_t minAlignment;

    float emptinessFactor;
    float traversalCost;
    float intersectionCost;
    uint32_t maxDepth;

    bool operator==(const Settings &) const = default;
    bool operator!=(const Settings &) const = default;

    void check() const;
};

}  // namespace builder
