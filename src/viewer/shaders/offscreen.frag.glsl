#version 460 core

#extension GL_GOOGLE_include_directive : enable

#include "uniform_buffer.glsl"

layout(set = 2, binding = 0) uniform sampler2D display;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(push_constant, scalar) uniform PushConstants
{
    layout(offset = 64) float x;
} pushConstants;

void main()
{
    fragColor = vec4(texture(display, uv).rgb, uniformBuffer.alpha);
}
