#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_demote_to_helper_invocation : enable

#include "uniform_buffer.glsl"

layout(set = 1, binding = 0) uniform sampler2D display;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 color = texture(display, uv);
    if (color.a == 0.0f) {
        demote;
    } else {
        fragColor = vec4(color.rgb, uniformBuffer.alpha);
    }
}
