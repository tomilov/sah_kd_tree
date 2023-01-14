#version 460 core

layout(location = 0) in vec2 coords;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    float t;
} ubuf;

void main()
{
    float i = 1. - (pow(abs(coords.x), 4.) + pow(abs(coords.y), 4.));
    float t = ubuf.t;
    i = smoothstep(t - 0.8, t + 0.8, i);
    i = floor(i * 20.) / 20.;
    fragColor = vec4(coords * .5 + .5, i, i);
    //fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
