#pragma once

#include <compute/fwd.hpp>

#include <optional>

#include <compute/compute_export.h>

namespace compute
{

CudaDevicePtr makeCudaDevice(const std::optional<DeviceUuidType> & deviceUuid) COMPUTE_EXPORT;

}  // namespace compute
