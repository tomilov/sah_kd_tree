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
    explicit FastPimpl(Args &&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>)
    {
        new (static_cast<void *>(get())) T{std::forward<Args>(args)...};
    }

    FastPimpl(const FastPimpl & v) : FastPimpl{*v}
    {}

    FastPimpl(FastPimpl && v) noexcept(std::is_nothrow_move_constructible_v<T>) : FastPimpl{std::move(*v)}
    {}

    FastPimpl & operator=(const FastPimpl & rhs)
    {
        *get() = *rhs;
        return *this;
    }

    FastPimpl & operator=(FastPimpl && rhs) noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        *get() = std::move(*rhs);
        return *this;
    }

    ~FastPimpl()
    {
        [[maybe_unused]] Validate<sizeof(T), alignof(T)> validate;
        get()->~T();
    }

    T * get() noexcept
    {
        return std::launder(reinterpret_cast<T *>(&storage));
    }

    const T * get() const noexcept
    {
        return std::launder(reinterpret_cast<const T *>(&storage));
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
    alignas(kAlignment) std::byte storage[kSize];

    template<size_t kActualSize, size_t kActualAlignment>
    struct Validate
    {
        static_assert(kSize == kActualSize);
        static_assert(kAlignment == kActualAlignment);
    };
};

}  // namespace utils
