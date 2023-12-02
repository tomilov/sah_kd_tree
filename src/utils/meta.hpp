#pragma once

#include <type_traits>

namespace meta
{

template<typename>
inline constexpr bool kAlwaysFalse = false;

namespace impl
{

template<typename AlwaysVoid, template<typename...> typename Trait, typename... Args>
struct Detector
{
    static constexpr bool value = false;
};

template<template<typename...> typename Trait, typename... Args>
struct Detector<std::void_t<Trait<Args...>>, Trait, Args...>
{
    static constexpr bool value = true;
};

}  // namespace impl

template<template<typename...> typename Trait, typename... Args>
inline constexpr bool kIsDetected = impl::Detector<void, Trait, Args...>::value;

}  // namespace meta
