#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

namespace sah_kd_tree
{
template<typename Iterator>
using IteratorValueType = typename thrust::iterator_value<Iterator>::type;

template<typename Type>
struct doubler
{
    __host__ __device__ thrust::tuple<Type, Type> operator()(Type value) const
    {
        return {value, value};
    }
};
}  // namespace sah_kd_tree
