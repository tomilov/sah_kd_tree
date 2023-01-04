#pragma once

#include <utils/assert.hpp>

#include <memory>

#include <cstddef>

namespace utils
{

template<typename T>
class CheckedPtr
{
public:
    constexpr CheckedPtr(std::nullptr_t)
    {}

    constexpr CheckedPtr(T * p) : p{p}
    {
        INVARIANT(p, "Empty CheckedPtr");
        checked = true;
    }

    explicit constexpr operator bool() const noexcept
    {
        checked = true;
        return p;
    }

    T * get() const
    {
        ASSERT_MSG(checked, "CheckedPtr contents were not checked before dereferencing");
        INVARIANT(p, "Empty CheckedPtr");
        return p;
    }

    T * operator->() const
    {
        return get();
    }

    T & operator*() const
    {
        return *get();
    }

private:
    mutable bool checked = false;
    T * p = nullptr;
};

template<typename T>
CheckedPtr(T * p) -> CheckedPtr<T>;

}  // namespace utils
