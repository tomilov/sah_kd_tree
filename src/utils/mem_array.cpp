#include <utils/mem_array.hpp>

#include <type_traits>

#include <cstddef>

namespace utils
{

static_assert(!std::is_copy_constructible_v<MemArray<std::byte>>);
static_assert(!std::is_copy_assignable_v<MemArray<std::byte>>);
static_assert(std::is_nothrow_move_constructible_v<MemArray<std::byte>>);
static_assert(std::is_nothrow_move_assignable_v<MemArray<std::byte>>);
static_assert(std::is_nothrow_swappable_v<MemArray<std::byte>>);

}  // namespace utils
