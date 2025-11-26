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
    vec3 lt;
    vec3 rt;
    vec3 lb;
    vec3 rb;
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
    vec2 uv;
    vec3 normal;
    float t;
};

// TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
bool rayTriangleIntersectMoeller(const in Ray ray, const in Triangle triangle, out vec2 uv, out vec3 normal, out float t)
{
    const vec3 ca = triangle.a - triangle.c;
    const vec3 bc = triangle.c - triangle.b;
    const vec3 c = ray.pos - triangle.c;
    normal = cross(bc, ca);
    const float invPlaneDist = 1.0f / dot(ray.dir, normal);
    t = -dot(normal, c) * invPlaneDist;
    if (t <= 0.0f) {
        return false;
    }
    const vec3 q = cross(c, ray.dir);
    uv.x = dot(q, bc) * invPlaneDist;
    uv.y = dot(q, ca) * invPlaneDist;
    return all(lessThanEqual(vec3(-kEps), vec3(uv, 1.0f - uv.x - uv.y)));
}

vec3 stableTriangleNormal(const in vec3 a, const in vec3 b, const in vec3 c)
{
    const vec3 ab = vec3(a.z * b.y, a.x * b.z, a.y * b.x);
    const vec3 bc = vec3(b.z * c.y, b.x * c.z, b.y * c.x);
    const vec3 AB = vec3(a.y * b.z - ab.x, a.z * b.x - ab.y, a.x * b.y - ab.z);
    const vec3 BC = vec3(b.y * c.z - bc.x, b.z * c.x - bc.y, b.x * c.y - bc.z);
    return mix(BC, AB, lessThan(abs(ab), abs(bc)));
}

bool rayTriangleIntersectPluecker(const in Ray ray, const in Triangle triangle, out vec2 uv, out vec3 normal, out float t)
{
    const vec3 a = triangle.a - ray.pos;
    const vec3 b = triangle.b - ray.pos;
    const vec3 c = triangle.c - ray.pos;

    const vec3 x = c - b;
    const vec3 y = a - c;
    const vec3 z = b - a;

    uv = vec2(dot(cross(x, b + c), ray.dir), dot(cross(y, c + a), ray.dir));
    const float w = dot(cross(z, a + b), ray.dir);

    const float sum = uv.x + uv.y + w;
    const float eps = kUlp * abs(sum);
    if (any(lessThan(vec3(eps), vec3(uv, w))) && any(lessThan(vec3(uv, w), vec3(eps)))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = dot(normal, a) / dot(ray.dir, normal);
    if (t <= 0.0f) {
        return false;
    }
    uv = min(uv / sum, 1.0f);
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
            vec2 uv;
            vec3 normal;
            if (rayTriangleIntersect(ray, triangles.triangle[triangle], uv, normal, t)) {
                if (t < hit.t) {
                    hit.triangle = triangle;
                    hit.uv = uv;
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
    const vec3 topRight = frustum.rt - frustum.lb;
    const vec3 topLeft = frustum.lt - frustum.rb;
    // times 4 orts of camera space: right, up and forward
    return mat3(
        topRight - topLeft,
        topRight + topLeft,
        frustum.rt + frustum.lb + frustum.lt + frustum.rb
    );
}

// get xy components in clip space (z is not needed) and inverse of homogeneous w component of p given in world coordinates
vec3 projectToClip(vec3 p, const in mat3 viewMatrix)
{
    p -= pos;
    p *= viewMatrix;
    p.z = 1.0f / p.z;
    p.xy *= p.z;
    return p;
}

void getAnalyticBaryDeriv(const in Triangle triangle, const in vec3 p, const in vec2 uv, out vec2 ddu, out vec2 ddv)
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
    const float pw = 4.0f * dot(p, viewMatrix[2]);  // don't actually know why multiplying by 4 is needed
    ddu = pw * (ddu - uv.x * sum);
    ddv = pw * (ddv - uv.y * sum);
}

void getBaryDeriv(const in vec3 rayDir, const in Hit hit, const in vec2 invImageExtent, out vec2 ddx, out vec2 ddy)
{
    // there is no sense in branching for calculation of all combinations
    // of these two conditions, because all these would be executed in the same subgroup eventually
    if ((hit.triangle == subgroupQuadSwapHorizontal(hit.triangle)) && (hit.triangle == subgroupQuadSwapVertical(hit.triangle))) {
        ddx = dFdx(hit.uv);
        ddy = dFdy(hit.uv);
    } else {  // there are no counterparts in quad to correctly calculate perspective correct derivatives as differences
        vec2 ddu;
        vec2 ddv;
        getAnalyticBaryDeriv(triangles.triangle[hit.triangle], hit.t * rayDir, hit.uv, ddu, ddv);
        ddu *= invImageExtent.yx;
        ddv *= invImageExtent.yx;
        ddx = vec2(ddu.x, ddv.x);
        ddy = vec2(ddu.y, ddv.y);
    }
}

void main() [[maximally_reconverges]]
{
    const uvec2 imageExtent = uvec2(imageSize(target)); // wrong!
    if ((imageExtent.x <= gl_GlobalInvocationID.x) || (imageExtent.y <= gl_GlobalInvocationID.y)) {
        return;
    }
    const vec2 invImageExtent = 1.0f / imageExtent;
    const vec2 pixel = (gl_GlobalInvocationID.xy + 0.5f) * invImageExtent;
#if 1
    // affine
    const vec3 dir = mix(
        mix(
            frustum.lb,
            frustum.lt,
            pixel.y
        ),
        mix(
            frustum.rb,
            frustum.rt,
            pixel.y
        ),
        pixel.x
    );
#else
    // equiangularly (just for fun, because current partial derivatives are irrelevant)
    const float phi = acos(dot(normalize(frustum.lt), normalize(frustum.rt))) * pixel.x;
    const float sinPhi = sin(phi);
    const float cosPhi = cos(phi);
    const vec3 top = cross(frustum.lt, normalize(cross(frustum.rt, frustum.lt))) * sinPhi + frustum.lt * cosPhi;
    const vec3 bottom = cross(frustum.lb, normalize(cross(frustum.rb, frustum.lb))) * sinPhi + frustum.lb * cosPhi;
    const float theta = acos(dot(normalize(top), normalize(bottom))) * pixel.y;
    const vec3 dir = cross(bottom, normalize(cross(top, bottom))) * sin(theta) + bottom * cos(theta);
#endif
    Ray ray;
    ray.pos = pos;
    ray.dir = normalize(dir);
    Hit hit;
    hit.triangle = ~0u;
    hit.t = kInf;
    const bool isHit = traceRay(nodeIndex, ray, hit);
    vec4 color;
    if (isHit) {
        const vec3 uvw = vec3(hit.uv, 1.0f - (hit.uv.x + hit.uv.y));
        if (wireFrameThickness > 0.0f) {
            vec3 ddx;
            vec3 ddy;
            getBaryDeriv(ray.dir, hit, invImageExtent, ddx.xy, ddy.xy);
            ddx.z = -(ddx.x + ddx.y);
            ddy.z = -(ddy.x + ddy.y);
            color.rgb = getWireFrameIntensity(uvw, ddx, ddy, wireFrameThickness).sss;
        } else {
            color.rgb = uvw;
        }
        color.a = 1.0f;
    } else {
        color = clearColor;
    }
    if (any(isnan(color))) {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    imageStore(target, ivec2(gl_GlobalInvocationID.xy), color);
}

