#pragma once

#include <cstddef>

namespace utils
{

template<typename Dividend, typename Divisor>
[[nodiscard]] auto divUp(Dividend dividend, Divisor divisor)
{
    dividend += divisor - 1;
    return dividend / divisor;
}

template<typename Dividend, typename Divisor>
[[nodiscard]] auto modDown(Dividend dividend, Divisor divisor)
{
    dividend += divisor - 1;
    return dividend % divisor;
}

}  // namespace utils
