#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include <cstddef>

namespace utils
{

template<typename T, std::size_t kSize, std::size_t kAlignment>
class FastPimpl final {
public:
    template<typename ...Args>
    explicit FastPimpl(Args &&... args) noexcept(noexcept(T(std::declval<Args>()...)))
    {
        new (operator->()) T{std::forward<Args>(args)...};
    }

    FastPimpl(FastPimpl && v) noexcept(noexcept(T(std::declval<T>())))
        : FastPimpl{std::move(*v)} {}

    FastPimpl(const FastPimpl & v) noexcept(noexcept(T{std::declval<const T &>()}))
        : FastPimpl{*v} {}

    FastPimpl & operator = (const FastPimpl & rhs) noexcept(noexcept(std::declval<T &>() = std::declval<const T &>())) {
      operator*() = *rhs;
      return *this;
    }

    FastPimpl & operator = (FastPimpl && rhs) noexcept(noexcept(std::declval<T &>() = std::declval<T>())) {
      operator*() = std::move(*rhs);
      return *this;
    }

    ~FastPimpl() noexcept {
      [[maybe_unused]] Validate<sizeof(T), alignof(T)> validate;
      operator*().~T();
    }

    T * operator->() noexcept { return reinterpret_cast<T *>(&storage_); }
    const T * operator->() const noexcept { return reinterpret_cast<const T *>(&storage_); }

    T & operator*() noexcept { return *operator->(); }
    const T & operator*() const noexcept { return *operator->(); }

private:
    alignas(kAlignment) std::byte storage_[kSize];

    template <std::size_t kActualSize, std::size_t kActualAlignment>
    struct Validate
    {
      static_assert(kSize == kActualSize);
      static_assert(kAlignment % kActualAlignment == 0);
    };
};

}
