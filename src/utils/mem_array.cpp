#include <utils/mem_array.hpp>

#include <type_traits>

namespace utils
{

static_assert(!std::is_copy_constructible_v<MemArray<char>>);
static_assert(!std::is_copy_assignable_v<MemArray<char>>);
static_assert(std::is_nothrow_move_constructible_v<MemArray<char>>);
static_assert(std::is_nothrow_move_assignable_v<MemArray<char>>);

}  // namespace utils
