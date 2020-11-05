#pragma once

#include <sah_kd_tree/config.hpp>
#include <sah_kd_tree/types.hpp>

#include <thrust/system/cuda/pointer.h>

namespace SahKdTree
{
void build(const Params & sah, thrust::cuda::pointer<const Triangle> trianglesBegin, thrust::cuda::pointer<const Triangle> trianglesEnd);
}
