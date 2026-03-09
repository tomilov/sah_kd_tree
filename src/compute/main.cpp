#include <compute/make.hpp>

#include <optional>

#include <cstdlib>

using namespace compute;

int main()
{
    [[maybe_unused]] auto cudaDevice = makeCudaDevice(std::nullopt);
    return EXIT_SUCCESS;
}
