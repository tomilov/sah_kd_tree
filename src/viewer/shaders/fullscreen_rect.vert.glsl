#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

//#extension GL_EXT_debug_printf : enable

#include "uniform_buffer.glsl"

layout(location = 0) out vec2 outUv;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
    //debugPrintfEXT("%i\n", gl_VertexIndex);
    outUv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uniformBuffer.transform2D * outUv * 2.0f - 1.0f, 0.5f, 1.0f);
}
