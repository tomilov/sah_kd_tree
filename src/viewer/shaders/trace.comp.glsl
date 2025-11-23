#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable
#extension GL_NV_compute_shader_derivatives : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_EXT_maximal_reconvergence : enable

#include "utils.glsl"

#define sizeof(Type) (uint64_t(Type(uint64_t(0))+1))

layout(local_size_x_id = 0, local_size_y_id = 1, derivative_group_quadsNV) in;
layout(constant_id = 2) const float kEps = 1E-7f;
layout(constant_id = 3) const float kInf = +1.0f / +0.0f;

struct Triangle  // sizeof == 36
{
    vec3 a, b, c;
};

struct Node  // sizeof == 64
{
    vec3 aabbMin;
    vec3 aabbMax;
    uvec3 leftRope;
    uvec3 rightRope;
    int splitDimension;
    float splitPos;
    uint leftChild;
    uint rightChild;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Triangles
{
    Triangle triangle[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Polygons
{
    uint triangle[];
};

layout(buffer_reference, scalar, buffer_reference_align = 64) readonly buffer Nodes
{
    Node node[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NodeParents
{
    uint parent[];
};

struct Frustum
{
    vec3 leftTop;
    vec3 rightTop;
    vec3 leftBottom;
    vec3 rightBottom;
};

layout(set = 0, binding = 0, scalar) uniform TreeUniformBuffer
{
    uint triangleCount;
    uint treeDepthMax;
    uint polygonCount;
    uint nodeCount;
    Triangles triangles;
    Polygons polygons;
    Nodes nodes;
    NodeParents nodeParents;
};

layout(set = 1, binding = 0, rgba8) uniform image2D target;

struct Ray
{
    vec3 pos;
    vec3 dir;
};

struct Hit
{
    uint triangle;
    float t;
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

// TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
bool rayTriangleIntersect(const in Ray ray, const in Triangle triangle, out vec2 uv, out float t)
{
    const vec3 v1v0 = triangle.b - triangle.a;
    const vec3 v2v0 = triangle.c - triangle.a;
    const vec3 tVec = ray.pos - triangle.a;
    const vec3 normal = cross(v1v0, v2v0);
    const float denom = dot(ray.dir, normal);
    if (abs(denom) < kEps) {
        return false;
    }
    const float invDenom = 1.0f / denom;
    t = -dot(normal, tVec) * invDenom;
    if (t <= 0.0f) {
        return false;
    }
    const vec3 q = cross(tVec, ray.dir);
    uv.x = invDenom * dot(-q, v2v0);
    uv.y = invDenom * dot(q, v1v0);
    return !((uv.x < -kEps) || (uv.y < -kEps) || (uv.x + uv.y > 1.0f + kEps));
}

bool rayTriangleIntersect2(const in Node node, const in Ray ray, inout Hit hit, const in Triangle triangle, const in float tNear, const in float tFar)
{
    // Can be shared between all triangles
    vec3 centerU = ray.dir;
    vec3 centerV = cross(ray.dir, ray.pos);
    // Constant
    vec3 v0U = (triangle.b - triangle.a);
    vec3 v0V = cross(triangle.b, triangle.a);
    vec3 v1U = (triangle.c - triangle.b);
    vec3 v1V = cross(triangle.c, triangle.b);
    vec3 v2U = (triangle.a - triangle.c);
    vec3 v2V = cross(triangle.a, triangle.c);
    // 6 dot intersection test
    const float s0 = dot(v0U, centerV) + dot(v0V, centerU);
    const float s1 = dot(v1U, centerV) + dot(v1V, centerU);
    const float s2 = dot(v2U, centerV) + dot(v2V, centerU);
    if (((s0 >= 0.0f) && (s1 >= 0.0f) && (s2 >= 0.0f)) || ((s0 <= 0.0f) && (s1 <= 0.0f) && (s2 <= 0.0f))) {
        vec3 normal = cross(triangle.b - triangle.a, triangle.c - triangle.a);
        float d = dot(ray.dir, normal);
        if (abs(d) < kEps) {
            return false;
        }
        // normal = normalize(normal);
        hit.t = dot(triangle.a - ray.pos, normal) / d;
        return !(hit.t < tNear) && !(tFar < hit.t);
    }
    return false;
}

uint findNode(uint nodeIndex, const in Ray ray)
{
    Node node = nodes.node[nodeIndex];
    while (any(lessThan(ray.pos, node.aabbMin)) || any(lessThan(node.aabbMax, ray.pos))) {
        if (nodeIndex == 0u) {
            return 0u;
        }
        nodeIndex = nodeParents.parent[nodeIndex];
        node = nodes.node[nodeIndex];
    }
    for (;;) {
        const int splitDimension = node.splitDimension;
        if (splitDimension < 0) {
            return nodeIndex;
        }
        if (ray.pos[splitDimension] < node.splitPos) {
            nodeIndex = node.leftChild;
        } else {
            nodeIndex = node.rightChild;
        }
        node = nodes.node[nodeIndex];
    }
}

bool traceRay(uint nodeIndex, const in Ray ray, inout Hit hit)
{
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    Node node = nodes.node[nodeIndex];
    vec3 aabbHitT = (mix(node.aabbMin, node.aabbMax, corner) - ray.pos) * invDir;
    float tMin = max(aabbHitT.x, max(aabbHitT.y, aabbHitT.z));
    for (;;) {
        for (;;) {
            const int splitDimension = node.splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == ((node.splitPos - ray.pos[splitDimension]) * invDir[splitDimension] < tMin)) {
                nodeIndex = node.leftChild;
            } else {
                nodeIndex = node.rightChild;
            }
            node = nodes.node[nodeIndex];
        }
        aabbHitT = (mix(node.aabbMax, node.aabbMin, corner) - ray.pos) * invDir;
        const float tMax = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
        if (tMin > tMax) {
            break;
        }
        const uint polygonStart = node.leftChild;
        const uint polygonEnd = polygonStart + node.rightChild;
        for (uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
            const uint triangle = polygons.triangle[polygon];
            vec2 uv;
            float t;
            if (rayTriangleIntersect(ray, triangles.triangle[triangle], uv, t)) {
                if (t < hit.t) {
                    hit.triangle = triangle;
                    hit.t = t;
                    hit.uv = uv;
                }
            }
        }
        if (hit.t <= tMax) {
            return true;
        }
        tMin = tMax;
        const ivec3 indices = mix(ivec3(0), ivec3(0, 1, 2), equal(aabbHitT, vec3(tMax)));
        const int ropeDirection = max(indices.x, max(indices.y, indices.z));
        nodeIndex = corner[ropeDirection] ? node.leftRope[ropeDirection] : node.rightRope[ropeDirection];
        //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
        if (nodeIndex == 0u) {
            break;
        }
        node = nodes.node[nodeIndex];
    }
    return hit.triangle != ~0u;
}

layout(push_constant, scalar) uniform PushConstants
{
    vec4 clearColor;
    vec3 pos;
    uint nodeIndex;
    Frustum frustum;
    float wireFrameThickness;
};

void main() [[maximally_reconverges]]
{
    const uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    const ivec2 imageSize = imageSize(target);
    if ((imageSize.x <= pixelCoords.x) || (imageSize.y <= pixelCoords.y)) {
        return;
    }
    const vec2 loc = (pixelCoords + 0.5f) / imageSize;
    const vec3 dir = mix(
        mix(
            frustum.leftBottom,
            frustum.leftTop,
            loc.y
        ),
        mix(
            frustum.rightBottom,
            frustum.rightTop,
            loc.y
        ),
        loc.x
    );
    Ray ray;
    ray.pos = pos;
    ray.dir = normalize(dir);
    Hit hit;
    hit.triangle = ~0u;
    hit.t = kInf;
    const bool isHit = traceRay(findNode(nodeIndex, ray), ray, hit);
    if (wireFrameThickness > 0.0f) {
        subgroupBarrier();
    }
    vec4 color;
    if (isHit) {
        vec3 baryCoord = vec3(1.0f - (hit.uv.x + hit.uv.y), hit.uv);
        if (wireFrameThickness > 0.0f) {
            if (subgroupAll(true)) {
                color.rgb = getWireFrameIntensity(baryCoord, wireFrameThickness).sss;
            } else {
                // TODO: analytical derivatives
                color.rgb = vec3(1.0f, 0.0f, 0.0f);
            }
        } else {
            color.rgb = baryCoord;
        }
        color.a = 1.0f;
    } else {
        color = clearColor;
    }
    if (any(isnan(color))) {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    imageStore(target, ivec2(pixelCoords), color);
}
