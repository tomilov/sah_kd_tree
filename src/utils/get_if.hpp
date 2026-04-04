#pragma once

#include <utils/pp.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace utils
{

template<typename T>
concept PointerLike = requires { std::declval<T &>() ? std::addressof(*std::declval<T &&>()) : nullptr; };

template<typename Leaf>
constexpr auto * getIf(Leaf && leaf)
{
    if constexpr (PointerLike<Leaf>) {
        return leaf ? ::utils::getIf(*std::forward<Leaf>(leaf)) : nullptr;
    } else {
        return std::addressof(std::forward<Leaf>(leaf));
    }
}

template<
    typename Root,
    typename Head,
    typename... Tail>
constexpr auto * getIf(
    Root && root,
    Head && head,
    Tail &&... tail)
{
    if constexpr (PointerLike<Root>) {
        return root ? ::utils::getIf(*std::forward<Root>(root), std::forward<Head>(head), std::forward<Tail>(tail)...) : nullptr;
    } else {
        return ::utils::getIf(std::invoke(std::forward<Head>(head), std::forward<Root>(root)), std::forward<Tail>(tail)...);
    }
}

}  // namespace utils

#define SKT_GET_IF_MEMBER_ACCESSOR(head)                                                                \
    , ([&]<typename Root>(Root && root) constexpr -> auto && { return std::forward<Root>(root).head; })

#define SKT_GET_IF(first, ...) ::utils::getIf(first SKT_FOR_EACH(SKT_GET_IF_MEMBER_ACCESSOR, __VA_ARGS__))
