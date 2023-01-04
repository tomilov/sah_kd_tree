#pragma once

namespace utils
{

struct NonCopyable
{
    NonCopyable() = default;
    NonCopyable(const NonCopyable &) = delete;
    void operator=(const NonCopyable &) = delete;
    NonCopyable(NonCopyable &&) = delete;
    void operator=(NonCopyable &&) = delete;
};

}  // namespace utils
