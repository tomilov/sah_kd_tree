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

template<typename Derived>
struct OneTime
{
    OneTime() = default;
    OneTime(const OneTime &) = delete;
    OneTime & operator=(const OneTime &) = delete;
    OneTime(OneTime &&) noexcept = default;
    OneTime & operator=(OneTime &&) noexcept = delete;

    static constexpr void checkTraits()
    {
        static_assert(!std::is_copy_constructible_v<Derived>);
        static_assert(!std::is_copy_assignable_v<Derived>);
        static_assert(std::is_nothrow_move_constructible_v<Derived>);
        static_assert(!std::is_move_assignable_v<Derived>);
    }
};

template<typename Derived>
struct Copyable
{
    Copyable() = default;
    Copyable(const Copyable &) = default;
    Copyable & operator=(const Copyable &) = default;
    Copyable(Copyable &&) noexcept = default;
    Copyable & operator=(Copyable &&) noexcept = default;

    static constexpr void checkTraits()
    {
        static_assert(std::is_copy_constructible_v<Derived>);
        static_assert(std::is_copy_assignable_v<Derived>);
        static_assert(std::is_nothrow_move_constructible_v<Derived>);
        static_assert(std::is_nothrow_move_assignable_v<Derived>);
    }
};

}  // namespace utils
