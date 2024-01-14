#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : enable
#extension GL_EXT_scalar_block_layout : enable

#include "uniform_buffer.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(push_constant, scalar) uniform PushConstants
{
    mat3x4 viewTransform;
    layout(offset = 48) float x;
} pushConstants;

void main()
{
    float t = uniformBuffer.t + pushConstants.x;
    if (t < 0.0f) {
        fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    float i = 1.0f - (pow(abs(uv.x), 4.0f) + pow(abs(uv.y), 4.0f));
    i = smoothstep(t - 0.8f, t + 0.8f, i);
    i = floor(i * 20.0f) / 20.0f;
    fragColor.rgb = vec3(uv * 0.5f + 0.5f, i);
    fragColor.a = uniformBuffer.alpha;
}
