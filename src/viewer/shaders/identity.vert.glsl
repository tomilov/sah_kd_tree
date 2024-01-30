#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "uniform_buffer.glsl"

layout(location = 0) in vec3 vertexPosition;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(push_constant, scalar) uniform PushConstants
{
    mat3 transform2D;
} pushConstants;

layout(std140, set = 0, binding = 1) restrict readonly buffer TransformBuffer
{
    mat4 transforms[];
} transformBuffer;

void main()
{
    vec4 worldVertexPosition = transformBuffer.transforms[gl_InstanceIndex] * vec4(vertexPosition, 1.0f);
    gl_Position = uniformBuffer.mvp * worldVertexPosition;
}

