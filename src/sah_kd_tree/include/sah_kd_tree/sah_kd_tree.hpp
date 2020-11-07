#pragma once

#include <sah_kd_tree/config.hpp>
#include <sah_kd_tree/types.hpp>

#include <thrust/device_ptr.h>

namespace SahKdTree
{
void build(const Params & sah, thrust::device_ptr<const Triangle> trianglesBegin, thrust::device_ptr<const Triangle> trianglesEnd);
}
