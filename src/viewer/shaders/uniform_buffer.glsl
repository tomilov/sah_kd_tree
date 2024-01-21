#extension GL_EXT_scalar_block_layout : enable

layout(std140, set = 0, binding = 0, scalar) uniform UniformBuffer
{
    float alpha;
    mat3 orientation;
    float t;
} uniformBuffer;
