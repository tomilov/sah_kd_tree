#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

//#extension GL_EXT_debug_printf : enable

#include "uniform_buffer.glsl"

layout(location = 0) out vec2 outUv;

void main()
{
    outUv = vec2((gl_VertexIndex >> 1) & 1, gl_VertexIndex & 1);
    //debugPrintfEXT("%i %f %f\n", gl_VertexIndex, outUv.x, outUv.y);
    vec2 position = outUv * 2.0f - 1.0f;
    position.y = -position.y;
    gl_Position = uniformBuffer.transform2D * vec4(position, 0.0f, 1.0f);
}
