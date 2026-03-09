#pragma once

namespace utils
{

template<
    typename Dividend,
    typename Divisor>
[[nodiscard]] constexpr auto divUp(
    Dividend dividend,
    Divisor divisor)
{
    dividend += divisor - 1;
    return dividend / divisor;
}

template<
    typename Dividend,
    typename Divisor>
[[nodiscard]] constexpr auto modDown(
    Dividend dividend,
    Divisor divisor)
{
    dividend += divisor - 1;
    return dividend % divisor;
}

template<typename Address>
[[nodiscard]] constexpr auto alignUp(
    Address address,
    Address alignment)
{
    if ((alignment & (alignment - 1)) == 0) {
        --address;
        --alignment;
        address |= alignment;
        ++address;
        return address;
    } else {
        return modDown(address, alignment) * alignment;
    }
}

}  // namespace utils
