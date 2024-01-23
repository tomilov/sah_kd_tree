#extension GL_EXT_scalar_block_layout : enable

layout(std140, set = 0, binding = 0, scalar) uniform UniformBuffer
{
    float t;
    float alpha;
    mat4 mvp;
} uniformBuffer;
