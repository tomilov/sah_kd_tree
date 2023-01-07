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
    NonCopyable & operator=(const OnlyMoveable &) = delete;
    OnlyMoveable(OnlyMoveable &&) = default;
    OnlyMoveable & operator=(OnlyMoveable &&) = default;
};

}  // namespace utils
