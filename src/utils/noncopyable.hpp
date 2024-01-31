#pragma once

#include <type_traits>

namespace utils
{

struct NonCopyable
{
    NonCopyable() = default;
    NonCopyable(const NonCopyable &) = delete;
    NonCopyable & operator=(const NonCopyable &) = delete;
    NonCopyable(NonCopyable &&) = delete;
    NonCopyable & operator=(NonCopyable &&) = delete;
};

struct OneTime
{
    OneTime() = default;
    OneTime(const OneTime &) = delete;
    OneTime & operator=(const OneTime &) = delete;
    OneTime(OneTime &&) noexcept = default;
    OneTime & operator=(OneTime &&) noexcept = delete;
};

template<typename T>
inline constexpr bool kIsOneTime = !std::is_copy_constructible_v<T> && !std::is_copy_assignable_v<T> && std::is_nothrow_move_constructible_v<T> && !std::is_nothrow_move_assignable_v<T>;

}  // namespace utils
