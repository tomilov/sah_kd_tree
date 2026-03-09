#pragma once

#include <functional>
#include <tuple>
#include <utility>

#include <cstddef>

namespace utils
{

template<typename... Args>
size_t getHash(const Args &... args)
{
    return (std::hash<Args>{}(args) ^ ...);
}

template<typename T>
struct Hash;

template<typename... Args>
struct Hash<std::tuple<Args...>>
{
    [[nodiscard]] size_t operator()(const std::tuple<Args...> & value) const noexcept
    {
        return getTupleHash(value, std::index_sequence_for<Args...>{});
    }

private:
    template<size_t... Indices>
    [[nodiscard]] size_t getTupleHash(
        const std::tuple<Args...> & value,
        std::index_sequence<Indices...>) const noexcept
    {
        return getHash(std::get<Indices>(value)...);
    }
};

}  // namespace utils
