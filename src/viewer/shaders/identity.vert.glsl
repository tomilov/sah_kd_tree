#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 vertexPosition;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) out float y;

layout(push_constant, scalar) uniform PushConstants
{
    mat4 mvp;
} pushConstants;

layout(std140, set = 0, binding = 1) restrict readonly buffer TransformBuffer
{
    mat4 transforms[];
} transformBuffer;

void main()
{
    vec4 worldVertexPosition = transformBuffer.transforms[gl_InstanceIndex] * vec4(vertexPosition, 1.0f);
    vec4 screenVertexPosition = pushConstants.mvp * worldVertexPosition;
    gl_Position = screenVertexPosition;
    y = screenVertexPosition.y;
}

