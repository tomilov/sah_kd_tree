#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : enable

#include "uniform_buffer.glsl"

layout(location = 0) out vec4 fragColor;

float wireFrame(in vec3 baryCoord, in float thickness)
{
    vec3 dBaryCoordX = dFdxFine(baryCoord);
    vec3 dBaryCoordY = dFdyFine(baryCoord);
    vec3 dBaryCoord  = sqrt(dBaryCoordX * dBaryCoordX + dBaryCoordY * dBaryCoordY);

    vec3 dThickness = dBaryCoord * thickness;

    vec3 remap = step(dThickness, baryCoord);
    float closestEdge = min(min(remap.x, remap.y), remap.z);

    return closestEdge;
}

void main()
{
    vec3 baryCoord = gl_BaryCoordEXT;
    if (uniformBuffer.wireFrame) {
        fragColor.rgb = wireFrame(baryCoord, 1.0f).sss;
    } else {
        fragColor.rgb = baryCoord;
    }
    fragColor.a = 1.0f;
}
