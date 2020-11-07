#pragma once

#include <sah_kd_tree/config.hpp>

#include <thrust/device_vector.h>
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
template<I dimension>
struct Projection
{
    using X = Projection;
    using Y = Projection<(dimension + 1) % 3>;
    using Z = Projection<(dimension + 2) % 3>;

    struct
    {
        thrust::device_vector<F> a, b, c;
    } triangle;

    struct
    {
        thrust::device_vector<F> min, max;
    } polygon;

    struct
    {
        thrust::device_vector<F> min, max;
    } node;

    struct
    {
        thrust::device_vector<U> node;
        thrust::device_vector<F> pos;
        thrust::device_vector<I> kind;
        thrust::device_vector<U> polygon;
    } event;

    void calculateTriangleBbox();
    void caluculateRootNodeBbox();
    void generateInitialEvent();
};

struct Builder
{
    Projection<0> x;
    Projection<1> y;
    Projection<2> z;

    void operator()(const Params & sah);
};

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
