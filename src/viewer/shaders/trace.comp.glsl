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
    float distance;
    uint triangle;
    vec2 uv;
};

float clearZeroSign(const in float x)
{
    return floatBitsToUint(x) == 0x80000000u ? 0.0f : x;
}

vec3 clearZeroSign(const in vec3 v)
{
    return vec3(clearZeroSign(v.x), clearZeroSign(v.y), clearZeroSign(v.z));
}

bool rayTriangleIntersection(const in Tree tree, const in Ray ray, inout Hit hit, const in float tNear, const in float tFar)
{
    // TODO:
    return true;
}

bool traceRay(const in Tree tree, in uint nodeIndex, const in Ray ray, inout Hit hit, out vec4 color)
{
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    vec3 aabbMin = vec3(tree.x.min[nodeIndex], tree.y.min[nodeIndex], tree.z.min[nodeIndex]);  // TODO: SoA to AoS for that (ProjectionNode.xyz.min -> ProjectionNode.min.xyz)
    vec3 aabbMax = vec3(tree.x.max[nodeIndex], tree.y.max[nodeIndex], tree.z.max[nodeIndex]);  // TODO: SoA to AoS for that (ProjectionNode.xyz.max -> ProjectionNode.max.xyz)
    vec3 aabbHitDistances = (mix(aabbMin, aabbMax, corner) - ray.src) * invDir;
    float tMin = min(aabbHitDistances.x, min(aabbHitDistances.y, aabbHitDistances.z));
    // TODO: walk down, then walk using ropes
    do {
        const float tNear = min(0.0f, tMin);
        for (;;) {
            const int splitDimension = tree.node.splitDimension[nodeIndex];
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == (tree.node.splitPos[nodeIndex] - ray.src[splitDimension]) * invDir[splitDimension] < tNear) {
                nodeIndex = tree.node.leftChild[nodeIndex];
            } else {
                nodeIndex = tree.node.rightChild[nodeIndex];
            }
        }
        aabbMin = vec3(tree.x.min[nodeIndex], tree.y.min[nodeIndex], tree.z.min[nodeIndex]);
        aabbMax = vec3(tree.x.max[nodeIndex], tree.y.max[nodeIndex], tree.z.max[nodeIndex]);
        aabbHitDistances = (mix(aabbMin, aabbMax, corner) - ray.src) * invDir;
        const float tMax = min(aabbHitDistances.x, min(aabbHitDistances.y, aabbHitDistances.z));
        if (tMin > tMax) {
            break;
        }
        const uvec3 indices = uvec3(equal(aabbHitDistances, vec3(tMax)));
        const uint ropeDirection = findLSB((indices.x << 0) | (indices.y << 1) | (indices.z << 2));  // rope direction
        const uint polygonStart = tree.node.leftChild[nodeIndex];
        const uint polygonEnd = polygonStart + tree.node.rightChild[nodeIndex];
        for (uint t = polygonStart; t < polygonEnd; ++t) {
            float tFar = min(hit.distance, tMax);
            Hit newHit;
            newHit.triangle = tree.polygon.triangle[t];
            if (rayTriangleIntersection(tree, ray, newHit, tNear, tFar)) {
                hit = newHit;
            }
        }
        if (hit.distance <= tMax) {
            return true;
        }
        tMin = tMax;
        nodeIndex = /* TODO: tree.node.ropes[nodeIndex][ropeDirection] */ 0;
    } while (nodeIndex != 0);
    return false;
}

layout(local_size_x = 32, local_size_y = 32) in;
layout(local_size_x_id = 3) in;
layout(local_size_y_id = 4) in;

layout(binding = 0, rgba8) uniform image2D colorImage;

struct Frustum
{
    vec3 leftTop, rightTop, leftBottom, rightBottom;
};

layout(push_constant, scalar) uniform PushConstants
{
    Tree tree;
    uint nodeIndex;
    vec3 pos;
    Frustum frustum;
    vec4 clearColor;
};

void main()
{
    uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    vec2 loc = (pixelCoords + 0.5f) / (gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    Ray ray;
    ray.src = pos;
    ray.dir = mix(mix(frustum.leftBottom, frustum.rightBottom, loc.x), mix(frustum.leftTop, frustum.rightTop, loc.x), loc.y);
    Hit hit;
    hit.distance = 0.0f;
    vec4 color;
    if (traceRay(tree, nodeIndex, ray, hit, color)) {
        color = vec4(1.0f - hit.uv.x - hit.uv.y, hit.uv, 1.0f);
    } else {
        color = clearColor;
    }
    imageStore(colorImage, ivec2(pixelCoords), color);
}
