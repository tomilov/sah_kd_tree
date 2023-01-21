#version 460 core

#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec2 vertices;
layout(location = 0) out vec2 coords;

out gl_PerVertex { vec4 gl_Position; };

layout(push_constant, scalar) uniform PushConstants {
    mat4 viewTransform;
} pushConstants;

void main()
{
    gl_Position = pushConstants.viewTransform * vec4(vertices, 0.0f, 1.0f);
    coords = vertices;
}

