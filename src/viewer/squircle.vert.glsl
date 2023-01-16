#version 460 core

layout(location = 0) in vec2 vertices;
layout(location = 0) out vec2 coords;

out gl_PerVertex { vec4 gl_Position; };

layout(std140, set = 0, binding = 0) uniform UniformBuffer
{
    float z;
    float alpha;

    float t;
} uniformBuffer;

void main()
{
    gl_Position = vec4(vertices, (gl_VertexIndex == 0 || gl_VertexIndex == 2) ? uniformBuffer.z : 0.0f, 1.0f);
    coords = vertices;
}

