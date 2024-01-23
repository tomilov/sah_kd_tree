#pragma once

#include <utils/noncopyable.hpp>

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <utils/utils_export.h>

namespace utils
{

template<typename F, typename... Args>
class ScopeGuard : NonCopyable
{
    using Storage = std::pair<F, std::tuple<Args...>>;

public:
    ScopeGuard(F && f, Args &&... args) noexcept(std::is_nothrow_move_constructible_v<Storage>)  // NOLINT(google-explicit-constructor)
        : storage{std::forward<F>(f), std::forward_as_tuple(std::forward<Args>(args)...)}
    {}

    ScopeGuard(ScopeGuard && other) noexcept(std::is_nothrow_move_constructible_v<Storage>) : storage{std::move(other).storage}, isActive{std::exchange(other.isActive, false)}
    {}

    ~ScopeGuard() noexcept
    {
        if (isActive) {
            std::apply(storage.first, storage.second);
        }
    }

    void release()
    {
        isActive = false;
    }

private:
    Storage storage;
    bool isActive = true;
};

template<typename F, typename... Args>
ScopeGuard(F && f, Args &&... args) -> ScopeGuard<F, Args...>;

}  // namespace utils
