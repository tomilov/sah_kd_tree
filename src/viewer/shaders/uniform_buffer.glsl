#extension GL_EXT_scalar_block_layout : enable

layout(std140, set = 1, binding = 0, scalar) uniform UniformBuffer
{
    mat2 transform2D;
    float alpha;
    float zNear;
    float zFar;
    vec3 pos;
    float t;
} uniformBuffer;
