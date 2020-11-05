#pragma once

#include <sah_kd_tree/config.hpp>
#include <sah_kd_tree/types.hpp>

#include <thrust/system/cpp/pointer.h>

namespace SahKdTree
{
void build(const Params & sah, thrust::cpp::pointer<const Triangle> trianglesBegin, thrust::cpp::pointer<const Triangle> trianglesEnd);
}
