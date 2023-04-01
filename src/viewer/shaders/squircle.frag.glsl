#version 460 core

#extension GL_EXT_fragment_shader_barycentric : enable

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(std140, set = 0, binding = 0) uniform UniformBuffer
{
    float alpha;

    float t;
} uniformBuffer;

void main()
{
    if (uniformBuffer.t < 0.0f) {
        fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    float i = 1.0f - (pow(abs(uv.x), 4.0f) + pow(abs(uv.y), 4.0f));
    i = smoothstep(uniformBuffer.t - 0.8f, uniformBuffer.t + 0.8f, i);
    i = floor(i * 20.0f) / 20.0f;
    fragColor.rgb = vec3(uv * 0.5f + 0.5f, i);
    fragColor.a = uniformBuffer.alpha;
    fragColor.rgb = gl_BaryCoordEXT.rgb;
}
