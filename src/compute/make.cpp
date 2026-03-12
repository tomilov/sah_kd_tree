#include <compute/compute.hpp>
#include <compute/make.hpp>

#include <memory>

namespace compute
{

CudaDevicePtr makeCudaDevice(const std::optional<DeviceUuidType> & deviceUuid)
{
    return std::make_shared<CudaDevice>(deviceUuid ? CudaDevice{deviceUuid.value()} : CudaDevice{});
}

}  // namespace compute
