#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : enable

#include "uniform_buffer.glsl"
#include "utils.glsl"

layout(location = 0) out vec4 fragColor;

void main()
{
    vec3 baryCoord = gl_BaryCoordEXT;
    if (uniformBuffer.wireFrameThickness > 0.0f) {
        fragColor.rgb = getWireFrameIntensity(baryCoord, uniformBuffer.wireFrameThickness).sss;
    } else {
        fragColor.rgb = baryCoord;
    }
    fragColor.a = 1.0f;
}
