#pragma once

#include <type_traits>

namespace meta
{

struct NotDetected
{
};

namespace impl
{

template<typename Default, typename AlwaysVolid, template<typename...> typename Trait, typename... Args>
struct Detector
{
    using type = Default;
};

template<typename Default, template<typename...> typename Trait, typename... Args>
struct Detector<Default, std::void_t<Trait<Args...>>, Trait, Args...>
{
    using type = Trait<Args...>;
};

}  // namespace impl

// Checks whether a trait is correct for the given template args
template<template<typename...> typename Trait, typename... Args>
inline constexpr bool kIsDetected = !std::is_same_v<typename impl::Detector<NotDetected, void, Trait, Args...>::type, NotDetected>;

}  // namespace meta
