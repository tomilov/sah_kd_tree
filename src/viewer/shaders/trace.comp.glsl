#version 460 core

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
//#extension GL_EXT_debug_printf : enable
#extension GL_NV_compute_shader_derivatives : enable
#extension GL_KHR_shader_subgroup_quad : enable  // subgroupQuadSwapHorizontal
#extension GL_EXT_maximal_reconvergence : enable

#include "utils.glsl"

#define sizeof(Type) (uint64_t(Type(uint64_t(0))+1))

layout(local_size_x_id = 0, local_size_y_id = 1, derivative_group_quadsNV) in;
layout(constant_id = 2) const float kUlp = 1.175494e-38f;
layout(constant_id = 3) const float kEps = 1.192093e-7f;
layout(constant_id = 4) const float kInf = +1.0f / +0.0f;

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
    vec3 uvw;
    vec3 normal;
    float t;
};

// TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
bool rayTriangleIntersectMoeller(const in Ray ray, const in Triangle triangle, out vec3 uvw, out vec3 normal, out float t)
{
    const vec3 v1v0 = triangle.b - triangle.a;
    const vec3 v2v0 = triangle.c - triangle.a;
    const vec3 tVec = ray.pos - triangle.a;
    normal = cross(v1v0, v2v0);
    const float invDenom = 1.0f / dot(ray.dir, normal);
    t = -dot(normal, tVec) * invDenom;
    if (t <= 0.0f) {
        return false;
    }
    const vec3 q = cross(tVec, ray.dir);
    uvw.y = invDenom * dot(-q, v2v0);
    uvw.z = invDenom * dot(q, v1v0);
    uvw.x = 1.0f - uvw.y - uvw.z;
    return all(lessThanEqual(vec3(-kEps), uvw));
}

vec3 stableTriangleNormal(const in vec3 a, const in vec3 b, const in vec3 c)
{
    const vec3 ab = vec3(a.z * b.y, a.x * b.z, a.y * b.x);
    const vec3 bc = vec3(b.z * c.y, b.x * c.z, b.y * c.x);
    const vec3 AB = vec3(a.y * b.z - ab.x, a.z * b.x - ab.y, a.x * b.y - ab.z);
    const vec3 BC = vec3(b.y * c.z - bc.x, b.z * c.x - bc.y, b.x * c.y - bc.z);
    return mix(BC, AB, lessThan(abs(ab), abs(bc)));
}

bool rayTriangleIntersectPluecker(const in Ray ray, const in Triangle triangle, out vec3 uvw, out vec3 normal, out float t)
{
    const vec3 a = triangle.a - ray.pos;
    const vec3 b = triangle.b - ray.pos;
    const vec3 c = triangle.c - ray.pos;

    const vec3 x = c - b;
    const vec3 y = a - c;
    const vec3 z = b - a;

    uvw = vec3(dot(cross(x, b + c), ray.dir), dot(cross(y, c + a), ray.dir), dot(cross(z, a + b), ray.dir));

    const float sum = uvw.x + uvw.y + uvw.z;
    const float eps = kUlp * abs(sum);
    if (any(lessThan(vec3(eps), uvw)) && any(lessThan(uvw, vec3(eps)))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = dot(normal, a) / dot(ray.dir, normal);
    if (t <= 0.0f) {
        return false;
    }
    uvw = min(uvw / sum, 1.0f);
    return true;
}

#if 0
#define rayTriangleIntersect rayTriangleIntersectPluecker
#else
#define rayTriangleIntersect rayTriangleIntersectMoeller
#endif

uint findNode(uint nodeIndex, const in vec3 rayPos)
{
    while (nodeIndex != 0u) {
        Node node = nodes.node[nodeIndex];
        if (all(lessThanEqual(node.aabbMin, rayPos)) && all(lessThanEqual(rayPos, node.aabbMax))) {
            break;
        }
        nodeIndex = nodeParents.parent[nodeIndex];
    }
    return nodeIndex;
}

// https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
bool traceRay(uint nodeIndex, const in Ray ray, inout Hit hit)
{
    const vec3 invDir = 1.0f / ray.dir;
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    Node node = nodes.node[nodeIndex];
    vec3 aabbHitT = (mix(node.aabbMin, node.aabbMax, corner) - ray.pos) * invDir;
    float t = max(aabbHitT.x, max(aabbHitT.y, aabbHitT.z));
    for (;;) {
        while (!(node.splitDimension < 0)) {
            if (corner[node.splitDimension] == ((node.splitPos - ray.pos[node.splitDimension]) * invDir[node.splitDimension] < t)) {
                nodeIndex = node.leftChild;
            } else {
                nodeIndex = node.rightChild;
            }
            node = nodes.node[nodeIndex];
        }
        const uint polygonStart = node.leftChild;
        const uint polygonEnd = polygonStart + node.rightChild;
        for (uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
            const uint triangle = polygons.triangle[polygon];
            vec3 uvw;
            vec3 normal;
            if (rayTriangleIntersect(ray, triangles.triangle[triangle], uvw, normal, t)) {
                if (t < hit.t) {
                    hit.triangle = triangle;
                    hit.uvw = uvw;
                    hit.normal = normal;
                    hit.t = t;
                }
            }
        }
        aabbHitT = (mix(node.aabbMax, node.aabbMin, corner) - ray.pos) * invDir;
        t = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
        if (hit.t <= t) {
            return true;
        }
        const ivec3 indices = mix(ivec3(0), ivec3(0, 1, 2), equal(aabbHitT, vec3(t)));
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
    uvec2 imageExtent_;  // TODO(tomilov): deal with viewport, scissor, width, height, actual image extent, etc correctly
    float wireFrameThickness;
};

mat3 getViewMatrix()
{
    const vec3 topRight = frustum.rightTop - frustum.leftBottom;
    const vec3 topLeft = frustum.leftTop - frustum.rightBottom;
    // times 4 orts of camera space: right, up and forward
    return mat3(
        topRight - topLeft,
        topRight + topLeft,
        frustum.rightTop + frustum.leftBottom + frustum.leftTop + frustum.rightBottom
    );
}

// get xy components in clip space (z is not needed) and inverse of homogeneous w component of p
vec3 projectToClip(vec3 p, const in mat3 viewMatrix)
{
    p -= pos;
    p *= viewMatrix;
    p.z = 1.0f / p.z;
    p.xy *= p.z;
    return p;
}

void getAnalyticBaryDeriv(const in Triangle triangle, const in vec3 p, const in vec3 uvw, out vec2 ddu, out vec2 ddv)
{
    const mat3 viewMatrix = getViewMatrix();
    const vec3 a = projectToClip(triangle.a, viewMatrix);
    const vec3 b = projectToClip(triangle.b, viewMatrix);
    const vec3 c = projectToClip(triangle.c, viewMatrix);
    const float invArea = 1.0f / ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    ddu = vec2(b.y - c.y, c.x - b.x) * (invArea * a.z);
    ddv = vec2(c.y - a.y, a.x - c.x) * (invArea * b.z);
    const vec2 ddw = vec2(a.y - b.y, b.x - a.x) * (invArea * c.z);
    const vec2 sum = ddu + ddv + ddw;
    const float pw = 4.0f * dot(p, viewMatrix[2]);  // multiplier accounts x4 ort lengths
    ddu = pw * (ddu - uvw.x * sum);
    ddv = pw * (ddv - uvw.y * sum);
}

void getBaryDeriv(const in vec3 rayDir, const in Hit hit, const in vec2 invImageExtent, out vec3 ddx, out vec3 ddy)
{
    if ((hit.triangle == subgroupQuadSwapHorizontal(hit.triangle)) && (hit.triangle == subgroupQuadSwapVertical(hit.triangle))) {
        ddx = dFdx(hit.uvw);
        ddy = dFdy(hit.uvw);
    } else {  // there are no counterparts in quad to correctly calculate derivatives as differences
        vec2 ddu;
        vec2 ddv;
        getAnalyticBaryDeriv(triangles.triangle[hit.triangle], hit.t * rayDir, hit.uvw, ddu, ddv);
        ddu *= invImageExtent.yx;
        ddv *= invImageExtent.yx;
        const vec2 ddw = -(ddu + ddv);
        ddx = vec3(ddu.x, ddv.x, ddw.x);
        ddy = vec3(ddu.y, ddv.y, ddw.y);
    }
}

void main() [[maximally_reconverges]]
{
    const uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    const uvec2 imageExtent = uvec2(imageSize(target)); // wrong!
    if ((imageExtent.x <= pixelCoords.x) || (imageExtent.y <= pixelCoords.y)) {
        return;
    }
    const vec2 invImageExtent = 1.0f / imageExtent;
    const vec2 pixel = (pixelCoords + 0.5f) * invImageExtent;
    const vec3 dir = mix(
        mix(
            frustum.leftBottom,
            frustum.leftTop,
            pixel.y
        ),
        mix(
            frustum.rightBottom,
            frustum.rightTop,
            pixel.y
        ),
        pixel.x
    );
    Ray ray;
    ray.pos = pos;
    ray.dir = normalize(dir);
    Hit hit;
    hit.triangle = ~0u;
    hit.t = kInf;
    const bool isHit = traceRay(nodeIndex, ray, hit);
    vec4 color;
    if (isHit) {
        if (wireFrameThickness > 0.0f) {
            vec3 ddx;
            vec3 ddy;
            getBaryDeriv(ray.dir, hit, invImageExtent, ddx, ddy);
            color.rgb = getWireFrameIntensity(hit.uvw, ddx, ddy, wireFrameThickness).sss;
        } else {
            color.rgb = hit.uvw;
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

