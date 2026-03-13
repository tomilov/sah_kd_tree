#pragma once

#include <type_traits>
#include <utility>

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

    struct CheckTraits;
    struct CheckTraitsThrow;

private:
    friend void swap(
        Derived & lhs,
        Derived & rhs) noexcept
    {
        static_assert(std::is_nothrow_swappable_v<Derived>);
        using std::swap;
        lhs.swap(rhs);
    }
};

template<typename Derived>
struct OneTime<Derived>::CheckTraits
{
    static_assert(!std::is_copy_constructible_v<Derived>);
    static_assert(!std::is_copy_assignable_v<Derived>);
    static_assert(std::is_nothrow_move_constructible_v<Derived>);
    static_assert(!std::is_move_assignable_v<Derived>);
    static_assert(std::is_nothrow_destructible_v<Derived>);
};

template<typename Derived>
struct OneTime<Derived>::CheckTraitsThrow
{
    static_assert(!std::is_copy_constructible_v<Derived>);
    static_assert(!std::is_copy_assignable_v<Derived>);
    static_assert(!std::is_nothrow_move_constructible_v<Derived> && std::is_move_constructible_v<Derived>);
    static_assert(!std::is_move_assignable_v<Derived>);
    static_assert(std::is_nothrow_destructible_v<Derived>);
};

template<typename Derived>
struct Copyable
{
    Copyable() = default;
    Copyable(const Copyable &) = default;
    Copyable & operator=(const Copyable &) = default;
    Copyable(Copyable &&) noexcept = default;
    Copyable & operator=(Copyable &&) noexcept = default;

    struct CheckTraits;
};

template<typename Derived>
struct Copyable<Derived>::CheckTraits
{
    static_assert(std::is_copy_constructible_v<Derived>);
    static_assert(std::is_copy_assignable_v<Derived>);
    static_assert(std::is_nothrow_move_constructible_v<Derived>);
    static_assert(std::is_nothrow_move_assignable_v<Derived>);
};

}  // namespace utils
