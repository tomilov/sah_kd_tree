#version 460 core

layout(location = 0) in vec2 coords;
layout(location = 0) out vec4 fragColor;

layout(std140, set = 0, binding = 0) uniform UniformBuffer {
    float t;
} uniformBuffer;

void main()
{
    float i = 1. - (pow(abs(coords.x), 4.) + pow(abs(coords.y), 4.));
    i = smoothstep(uniformBuffer.t - 0.8, uniformBuffer.t + 0.8, i);
    i = floor(i * 20.) / 20.;
    fragColor = vec4(coords * .5 + .5, i, 1.0);
}
