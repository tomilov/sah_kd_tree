#include <sah_kd_tree_fd/settings.hpp>
#include <utils/assert.hpp>

#include <limits>
#include <bit>

namespace sah_kd_tree_fd
{

void Settings::check() const
{
    ASSERT(!std::empty(deviceUuid));
    ASSERT(std::has_single_bit(minAlignment));
    ASSERT(emptinessFactor > 0.0f);
    ASSERT(traversalCost > 0.0f);
    ASSERT(intersectionCost > 0.0f);
    ASSERT(maxDepth < std::numeric_limits<uint32_t>::max());
}

}
