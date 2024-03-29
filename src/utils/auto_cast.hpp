#pragma once

#include <utils/assert.hpp>

#include <bit>
#include <limits>
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
        return static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs);
    } else if constexpr (std::is_signed_v<L>) {
        return (lhs < static_cast<L>(0)) || (static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs));
    } else {
        static_assert(std::is_signed_v<R>);
        return !(rhs < static_cast<R>(0)) && (static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs));
    }
}

template<typename To, typename From>
constexpr bool inRange(const From & value) noexcept
{
    if constexpr (std::is_same_v<To, bool> && (std::is_arithmetic_v<From> || std::is_pointer_v<From>)) {
        return true;
    } else {
        return !isLess(value, (std::numeric_limits<To>::lowest)()) && !isLess((std::numeric_limits<To>::max)(), value);
    }
}

template<typename To, typename From>
constexpr To convertIfInRange(From && value)
{
    INVARIANT(inRange<To>(value), "Unable to convert");
    return static_cast<To>(std::forward<From>(value));
}

template<typename Source>
class autoCast
{
public:
    constexpr explicit autoCast(Source && source) noexcept : source{source}
    {}

    template<typename Destination>
    constexpr operator Destination() const &&  // NOLINT: google-explicit-constructor
    {
        using S = std::remove_reference_t<Source>;
        if constexpr (std::is_same_v<S, Destination>) {
            return std::forward<Source>(source);
        } else if constexpr (std::is_enum_v<S>) {
            if constexpr (std::is_enum_v<Destination>) {
                using SourceUnderlyingType = std::underlying_type_t<S>;
                using DestinationUnderlyingType = std::underlying_type_t<Destination>;
                return static_cast<Destination>(convertIfInRange<DestinationUnderlyingType>(static_cast<SourceUnderlyingType>(source)));
            } else if constexpr (std::is_arithmetic_v<Destination>) {
                static_assert(!std::is_same_v<Destination, bool>);
                return convertIfInRange<Destination>(source);
            } else {
                static_assert(!std::is_pointer_v<Destination>);
                return static_cast<Destination>(source);
            }
        } else if constexpr (std::is_arithmetic_v<S>) {
            if constexpr (std::is_enum_v<Destination>) {
                static_assert(!std::is_same_v<S, bool>);
                return static_cast<Destination>(convertIfInRange<std::underlying_type_t<Destination>>(source));
            } else if constexpr (std::is_arithmetic_v<Destination>) {
                static_assert(!std::is_same_v<S, bool>);
                return convertIfInRange<Destination>(source);
            } else if constexpr (std::is_pointer_v<Destination>) {
                static_assert(!std::is_same_v<S, bool>);
                static_assert(!std::is_function_v<std::remove_pointer_t<Destination>>);
                return std::bit_cast<Destination>(convertIfInRange<uintptr_t>(source));
            } else {
                return static_cast<Destination>(source);
            }
        } else if constexpr (std::is_pointer_v<S>) {
            if constexpr (std::is_arithmetic_v<Destination>) {
                static_assert(!std::is_same_v<Destination, bool>);
                return convertIfInRange<Destination>(std::bit_cast<uintptr_t>(source));
            } else if constexpr (std::is_pointer_v<Destination>) {
                if constexpr (std::is_function_v<std::remove_pointer_t<S>>) {
                    static_assert(std::is_function_v<std::remove_pointer_t<Destination>>);
                    return std::bit_cast<Destination>(source);
                } else {
                    static_assert(!std::is_function_v<std::remove_pointer_t<Destination>>);
                    using Void = std::conditional_t<std::is_const_v<std::remove_pointer_t<S>>, std::add_const_t<void>, void>;
                    return static_cast<Destination>(static_cast<std::add_pointer_t<Void>>(source));
                }
            } else {
                static_assert(!std::is_enum_v<Destination>);
                return static_cast<Destination>(source);
            }
        } else {
            return static_cast<Destination>(std::forward<Source>(source));
        }
    }

private:
    Source & source;
};

template<typename Source>
autoCast(Source && source) -> autoCast<Source>;

template<typename Destination, typename Source>
constexpr Destination safeCast(Source && source)
{
    return autoCast<Source>{std::forward<Source>(source)}.operator Destination();
}

}  // namespace utils
