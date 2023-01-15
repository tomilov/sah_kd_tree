#version 460 core

layout(location = 0) in vec2 vertices;
layout(location = 0) out vec2 coords;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
    gl_Position = vec4(vertices, 0.0f, 1.0f);
    coords = vertices;
}

