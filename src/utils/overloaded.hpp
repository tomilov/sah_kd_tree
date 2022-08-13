#pragma once

namespace utils
{

template<typename... Ts>
struct Overloaded : Ts...
{
    using Ts::operator()...;
};

template<typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

}  // namespace utils
