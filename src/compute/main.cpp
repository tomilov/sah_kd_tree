#include <compute/make.hpp>

#include <optional>

#include <cstdlib>

using namespace compute;

int main()
{
    CudaDevicePtr cudaDevice = makeCudaDevice(std::nullopt);
    return EXIT_SUCCESS;
}
