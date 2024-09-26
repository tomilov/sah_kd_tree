#include <compute/compute.hpp>
#include <compute/make.hpp>

namespace compute
{

CudaDevicePtr makeCudaDevice(const std::optional<DeviceUuidType> & deviceUuid)
{
    return std::make_shared<CudaDevice>(deviceUuid);
}

}  // namespace compute
