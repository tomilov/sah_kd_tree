#version 460 core

#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 vertexPosition;

out gl_PerVertex { vec4 gl_Position; };

layout(push_constant, scalar) uniform PushConstants
{
    mat3x4 viewTransform;
} pushConstants;

layout(std140, set = 0, binding = 1) readonly buffer TransformBuffer
{
    mat4 transforms[];
} transformBuffer;

void main()
{
    vec4 worldVertexPosition = transformBuffer.transforms[gl_InstanceIndex] * vec4(vertexPosition, 1.0f);
    gl_Position.xyz = mat3(pushConstants.viewTransform) * worldVertexPosition.xyz;
    gl_Position.w = worldVertexPosition.w;
}

