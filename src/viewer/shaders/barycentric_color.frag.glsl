#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : enable

#include "uniform_buffer.glsl"

layout(location = 0) out vec4 fragColor;

void main()
{
    fragColor.rgb = gl_BaryCoordEXT.xyz;
    fragColor.a = uniformBuffer.alpha;
}
