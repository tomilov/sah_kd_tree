#pragma once

#include <../SPIRV-Reflect/spirv_reflect.h>

#include <string>

namespace engine
{

void dump(const spv_reflect::ShaderModule & shaderModule);

std::string serialize(const SpvReflectDescriptorSet & descriptorSet);

}  // namespace engine
