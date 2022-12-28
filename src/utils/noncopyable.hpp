#pragma once

#include <type_traits>

namespace utils
{

struct NonCopyable
{
    NonCopyable() = default;
    NonCopyable(const NonCopyable &) = delete;
    void operator=(const NonCopyable &) = delete;
};

struct NonMoveable
{
    NonMoveable() = default;
    NonMoveable(NonMoveable &&) = delete;
    void operator=(NonMoveable &&) = delete;
};

}  // namespace utils
