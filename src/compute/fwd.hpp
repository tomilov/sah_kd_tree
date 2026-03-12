#pragma once

#include <array>
#include <memory>

#include <cstddef>

#include <compute/compute_export.h>

namespace compute
{

using DeviceUuidType = std::array<std::byte, 16>;
class CudaDevice;
using CudaDevicePtr = std::shared_ptr<CudaDevice>;
class CudaStream;
class DeviceMemory;
class MappedDeviceMemory;

class CudaFileDriver;
class CudaFile;
class CudaFileReader;

}  // namespace compute
