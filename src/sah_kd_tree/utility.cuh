#pragma once

#include <thrust/tuple.h>

#include <chrono>
#include <iostream>
#include <string>
#include <utility>

struct Timer
{
    std::chrono::system_clock::time_point start = std::chrono::high_resolution_clock::now();

    void operator()(const std::string & description)
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::cout << description << " " << std::fixed << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - std::exchange(start, now)).count()) * 1E-9) << std::endl;
    }
};

namespace SahKdTree
{
template<typename Iterator>
using IteratorValueType = typename Iterator::value_type;

template<typename Type>
struct doubler
{
    __host__ __device__ thrust::tuple<Type, Type> operator()(Type value)
    {
        return {value, value};
    }
};
}  // namespace SahKdTree
