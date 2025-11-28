#version 460 core

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_scalar_block_layout: require

// #extension GL_EXT_debug_printf: require

#include "uniform_buffer.glsl"

layout(location = 0) out vec2 outUv;

void main()
{
    const vec2 size = vec2(uniformBuffer.width, uniformBuffer.height);
    outUv = (vec2((gl_VertexIndex >> 1) & 1, gl_VertexIndex & 1) * size + 0.5f) / (size + 1.0f);
    //debugPrintfEXT("%i %f %f\n", gl_VertexIndex, outUv.x, outUv.y);
    vec2 position = outUv * 2.0f - 1.0f;
    position.y = -position.y;
    gl_Position = uniformBuffer.windowMvp * vec4(position, 0.0f, 1.0f);
}
