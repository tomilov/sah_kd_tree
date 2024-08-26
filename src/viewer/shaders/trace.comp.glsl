#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference : require

#include "uniform_buffer.glsl"

layout(constant_id = 0) const uint kTriangleCount = 1;
layout(constant_id = 1) const uint kPolygonCount = 1;
layout(constant_id = 2) const uint kNodeCount = 1;

struct Triangle
{
    vec3 a, b, c;
};

struct ProjectionNode
{
    float min[kNodeCount], max[kNodeCount];
    uint leftRope[kNodeCount], rightRope[kNodeCount];
};

struct Polygon
{
    uint triangle[kPolygonCount];
};

struct Node  // TODO: transpose
{
    int splitDimension[kNodeCount];
    float splitPos[kNodeCount];
    uint leftChild[kNodeCount];
    uint rightChild[kNodeCount];
    uint parent[kNodeCount];
};

layout(scalar, buffer_reference, buffer_reference_align = 4) readonly buffer Tree
{
    Triangle triangles[kTriangleCount];
    ProjectionNode x, y, z;
    Polygon polygon;
    Node node;
};

struct Ray
{
    vec3 src;
    vec3 dir;
};

struct Hit
{
    uint nodeIndex;
    float distance;
    uint tirangleIndex;
    vec2 uv;
};

float clearZeroSign(float x)
{
    return floatBitsToUint(x) == 0x80000000u ? 0.0f : x;
}

vec3 clearZeroSign(vec3 v)
{
    return vec3(clearZeroSign(v.x), clearZeroSign(v.y), clearZeroSign(v.z));
}

struct Frustum
{
    vec3 leftTop, rightTop, leftBottom, rightBottom;
};

layout(push_constant, scalar) uniform PushConstants
{
    Tree tree;
    vec3 pos;
    Frustum frustum;
    uint nodeIndex;
    vec4 clearColor;
};

layout(local_size_x = 32, local_size_y = 32) in;
layout(local_size_x_id = 3) in;
layout(local_size_y_id = 4) in;

layout(binding = 0, rgba8) uniform image2D colorImage;

bool traceRay(inout Ray ray, inout Hit hit, out vec4 color)
{
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    vec3 aabbMin = vec3(tree.x.min[hit.nodeIndex], tree.y.min[hit.nodeIndex], tree.z.min[hit.nodeIndex]);  // TODO: Projection SoA to AoS for that
    vec3 aabbMax = vec3(tree.x.max[hit.nodeIndex], tree.y.max[hit.nodeIndex], tree.z.max[hit.nodeIndex]);  // TODO: Projection SoA to AoS for that
    vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    bvec3 sign = lessThan(invDir, vec3(0.0f));
    vec3 corner = (mix(aabbMin, aabbMax, sign) - ray.src) * invDir;
    float tMin = min(corner.x, min(corner.y, corner.z));
    // TODO: walk down, then walk using ropes

    color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
    return true;
}

void main()
{
    uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    vec2 loc = (pixelCoords + 0.5f) / (gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    Ray ray;
    ray.src = pos;
    ray.dir = mix(mix(frustum.leftBottom, frustum.rightBottom, loc.x), mix(frustum.leftTop, frustum.rightTop, loc.x), loc.y);
    Hit hit;
    hit.nodeIndex = nodeIndex;
    hit.distance = 0.0f;
    vec4 color;
    if (!traceRay(ray, hit, color)) {
        color = clearColor;
    }
    imageStore(colorImage, ivec2(pixelCoords), color);
}
