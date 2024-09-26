#pragma once

#include <array>
#include <memory>

#include <cstddef>

namespace compute
{

using DeviceUuidType = std::array<std::byte, 16>;

class MappedDeviceMemory;
class DeviceMemory;

class CudaDevice;
using CudaDevicePtr = std::shared_ptr<CudaDevice>;

}  // namespace compute
