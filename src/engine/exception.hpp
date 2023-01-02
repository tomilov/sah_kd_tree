#pragma once

#include <stdexcept>

namespace engine
{

class RuntimeError : public std::runtime_error
{
public:
    using runtime_error::runtime_error;
    ~RuntimeError() override = default;
};

}  // namespace engine
