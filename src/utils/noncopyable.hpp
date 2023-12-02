#pragma once

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

struct OnlyMoveable
{
    OnlyMoveable() = default;
    OnlyMoveable(const OnlyMoveable &) = delete;
    OnlyMoveable & operator=(const OnlyMoveable &) = delete;
    OnlyMoveable(OnlyMoveable &&) noexcept = default;
    OnlyMoveable & operator=(OnlyMoveable &&) noexcept = default;
};

}  // namespace utils
