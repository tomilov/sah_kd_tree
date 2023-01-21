#include <utils/random.hpp>

#include <random>

namespace utils
{

std::mt19937 & defaultRandom()
{
    thread_local std::mt19937 random;
    return random;
}

}  // namespace utils
