#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include <cstddef>

namespace utils
{

template<typename T, size_t kSize, size_t kAlignment>
class FastPimpl final
{
public:
    template<typename... Args>
    explicit FastPimpl(Args &&... args) noexcept(noexcept(T{std::declval<Args>()...}))
    {
        new (operator->()) T{std::forward<Args>(args)...};
    }

    FastPimpl(const FastPimpl & v) noexcept(noexcept(T{std::declval<const T &>()})) : FastPimpl{*v}
    {}

    FastPimpl(FastPimpl && v) noexcept(noexcept(T(std::declval<T>()))) : FastPimpl{std::move(*v)}
    {}

    FastPimpl & operator=(const FastPimpl & rhs) noexcept(noexcept(std::declval<T &>() = std::declval<const T &>()))
    {
        *get() = *rhs;
        return *this;
    }

    FastPimpl & operator=(FastPimpl && rhs) noexcept(noexcept(std::declval<T &>() = std::declval<T>()))
    {
        *get() = std::move(*rhs);
        return *this;
    }

    ~FastPimpl() noexcept
    {
        [[maybe_unused]] Validate<sizeof(T), alignof(T)> validate;
        get()->~T();
    }

    T * get() noexcept
    {
        return reinterpret_cast<T *>(&storage_);
    }

    const T * get() const noexcept
    {
        return reinterpret_cast<const T *>(&storage_);
    }

    T * operator->() noexcept
    {
        return get();
    }

    const T * operator->() const noexcept
    {
        return get();
    }

    T & operator*() noexcept
    {
        return *get();
    }

    const T & operator*() const noexcept
    {
        return *get();
    }

private:
    alignas(kAlignment) std::byte storage_[kSize];

    template<size_t kActualSize, size_t kActualAlignment>
    struct Validate
    {
        static_assert(kSize == kActualSize);
        static_assert(kAlignment % kActualAlignment == 0);
    };
};

}  // namespace utils
