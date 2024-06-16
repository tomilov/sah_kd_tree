#version 460 core

#extension GL_GOOGLE_include_directive : enable

#include "uniform_buffer.glsl"

layout(set = 1, binding = 0) uniform sampler2D display;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

void main()
{
    fragColor = vec4(texture(display, uv).rgb, uniformBuffer.alpha);
}
