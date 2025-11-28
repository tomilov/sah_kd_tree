//#pragma once

#extension GL_EXT_scalar_block_layout: require

layout(set = 0, binding = 0, scalar) uniform UniformBuffer
{
    bool useOffscreenTexture;
    bool discardInvisible;
    float wireframeThickness;
    vec3 position;
    float width;
    float height;
    float zNear;
    float zFar;
    float alpha;
    mat4 windowMvp;
} uniformBuffer;
