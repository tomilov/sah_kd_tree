#pragma once

#include <utils/assert.hpp>

#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include <cstdint>

namespace utils
{

template<typename L, typename R>
constexpr bool isLess(const L & lhs, const R & rhs) noexcept
{
    static_assert(std::is_arithmetic_v<L>);
    static_assert(!std::is_same_v<L, bool>);
    static_assert(std::is_arithmetic_v<R>);
    static_assert(!std::is_same_v<R, bool>);

    using CommonType = std::common_type_t<L, R>;
    if constexpr (std::is_signed_v<L> == std::is_signed_v<R>) {
        return lhs < rhs;
    } else if constexpr (std::is_signed_v<L> && !std::is_signed_v<R>) {
        return (lhs < static_cast<L>(0)) || (static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs));
    } else if constexpr (!std::is_signed_v<L> && std::is_signed_v<R>) {
        return !(rhs < static_cast<R>(0)) && (static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs));
    } else {
        static_assert(sizeof(CommonType) == 0, "Conversion is not possible");
    }
}

template<typename To, typename From>
constexpr bool isIncludes(const From & value) noexcept
{
    if constexpr (std::is_same_v<From, bool> || std::is_same_v<To, bool>) {
        return true;
    } else {
        return !isLess(value, (std::numeric_limits<To>::min)()) && !isLess((std::numeric_limits<To>::max)(), value);
    }
}

template<typename To, typename From>
constexpr std::optional<To> convertToOptionalIfIncludes(From && value)
{
    if (!isIncludes<To>(value)) {
        return std::nullopt;
    }
    return static_cast<To>(std::forward<From>(value));
}

template<typename To, typename From>
constexpr To convertIfIncludes(From && value)
{
    INVARIANT(isIncludes<To>(value), "Unable to convert");
    return static_cast<To>(std::forward<From>(value));
}

template<typename Source>
struct autoCast
{
    constexpr autoCast(Source && source) noexcept : source{source}
    {}

    template<typename Destination>
    constexpr operator Destination() const &&
    {
        if constexpr (std::is_arithmetic_v<Source> && std::is_arithmetic_v<Destination>) {
            return convertIfIncludes<Destination>(source);
        } else if constexpr (std::is_arithmetic_v<Source> && std::is_pointer_v<Destination>) {
            return reinterpret_cast<Destination>(convertIfIncludes<uintptr_t>(source));
        } else if constexpr (std::is_pointer_v<Source> && std::is_arithmetic_v<Destination>) {
            return convertIfIncludes<Destination>(reinterpret_cast<uintptr_t>(source));
        } else if constexpr (std::is_pointer_v<Source> && std::is_pointer_v<Destination>) {
            return reinterpret_cast<Destination>(source);
        } else {
            return static_cast<Destination>(std::forward<Source>(source));
        }
    }

private:
    Source & source;
};

template<typename Source>
autoCast(Source && source) -> autoCast<Source>;

}  // namespace utils
