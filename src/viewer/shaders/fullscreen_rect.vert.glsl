#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

//#extension GL_EXT_debug_printf : enable

#include "uniform_buffer.glsl"

layout(location = 0) out vec2 outUv;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
    vec2 uv;
    switch (gl_VertexIndex % 4) {
    case 0: {
        uv = vec2(-1.0f, -1.0f);
        break;
    }
    case 1: {
        uv = vec2(-1.0f, 1.0f);
        break;
    }
    case 2: {
        uv = vec2(1.0f, -1.0f);
        break;
    }
    case 3: {
        uv = vec2(1.0f, 1.0f);
        break;
    }
    }
    gl_Position = vec4(uniformBuffer.transform2D * uv, 0.0f, 1.0f);
    //debugPrintfEXT("%i\n", gl_VertexIndex);
    outUv = uv;
}

