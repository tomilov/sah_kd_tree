#pragma once

#include <stdexcept>

namespace renderer
{

class RuntimeError : public std::runtime_error
{
public:
    using runtime_error::runtime_error;
    ~RuntimeError() override = default;
};

}  // namespace renderer
