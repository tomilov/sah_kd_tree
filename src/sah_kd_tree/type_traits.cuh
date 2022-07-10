#pragma once

#include <thrust/iterator/iterator_traits.h>

namespace sah_kd_tree
{
template<typename Iterator>
using IteratorValueType = typename thrust::iterator_value<Iterator>::type;
}  // namespace sah_kd_tree
