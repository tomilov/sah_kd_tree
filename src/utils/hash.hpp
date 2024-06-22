#pragma once

#include <functional>

#include <cstddef>

namespace utils
{

template<typename... Args>
size_t getHash(const Args &... args)
{
    return (std::hash<Args>{}(args) ^ ...);
}

}  // namespace utils
