#include "sah_kd_tree.cuh"
#include "sah_kd_tree/sah_kd_tree.hpp"

namespace sah_kd_tree
{
template struct projection<0>;
template struct projection<1>;
template struct projection<2>;

void build()
{
    [[maybe_unused]] auto builder = sah_kd_tree::make_builder(thrust::cuda::par);
}
}  // namespace sah_kd_tree
