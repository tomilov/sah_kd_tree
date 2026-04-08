#version 460 core

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_NV_compute_shader_derivatives: require
#extension GL_KHR_shader_subgroup_quad: require  // subgroupQuadSwapHorizontal
#extension GL_EXT_maximal_reconvergence: require
#extension GL_EXT_expect_assume: require

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

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Indices
{
    uvec3 triangle[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vertices
{
    vec3 position[];
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
    Indices indices;
    Vertices vertices;
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
    uvw.x = dot(q, bc) * invPlaneDist;
    uvw.y = dot(q, ca) * invPlaneDist;
    uvw.z = 1.0f - (uvw.x + uvw.y);
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

    uvw = ray.dir * mat3(cross(x, b + c), cross(y, c + a), cross(z, a + b));

    const float sum = uvw.x + uvw.y + uvw.z;
    const float eps = kUlp * abs(sum);
    if (any(lessThan(uvw, vec3(-eps))) && any(lessThan(vec3(eps), uvw))) {
        return false;
    }
    normal = stableTriangleNormal(x, y, z);
    t = dot(a, normal) / dot(ray.dir, normal);
    if (t <= 0.0f) {
        return false;
    }
    uvw = min(uvw / sum, 1.0f);
    return true;
}

#if 1
#define rayTriangleIntersect rayTriangleIntersectPluecker
#else
#define rayTriangleIntersect rayTriangleIntersectMoeller
#endif

uint findNode(uint nodeIndex, const in vec3 pos)
{
    while (nodeIndex != 0u) {
        Node node = nodes.node[nodeIndex];
        if (all(lessThanEqual(node.aabbMin, pos)) && all(lessThanEqual(pos, node.aabbMax))) {
            break;
        }
        nodeIndex = nodeParents.parent[nodeIndex];
    }
    return nodeIndex;
}

Triangle getTriangle(uint t)
{
    const uvec3 index = indices.triangle[t];
    return Triangle(vertices.position[index.x], vertices.position[index.y], vertices.position[index.z]);
}

void traceRay(uint nodeIndex, const in Ray ray, inout Hit hit, float tMin)
{
    const vec3 invDir = 1.0f / ray.dir;
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    do {
        Node node = nodes.node[nodeIndex];
        while (expectEXT(!(node.splitDimension < 0), true)) {
            if (corner[node.splitDimension] == ((node.splitPos - ray.pos[node.splitDimension]) * invDir[node.splitDimension] < tMin)) {
                nodeIndex = node.leftChild;
            } else {
                nodeIndex = node.rightChild;
            }
            node = nodes.node[nodeIndex];
        }
        const uint polygonStart = node.leftChild;
        const uint polygonEnd = polygonStart + node.rightChild;
        for (uint polygon = polygonStart; expectEXT(polygon < polygonEnd, true); ++polygon) {
            const uint triangle = polygons.triangle[polygon];
            vec3 uvw;
            vec3 normal;
            if (rayTriangleIntersect(ray, getTriangle(triangle), uvw, normal, tMin)) {
                if (tMin < hit.t) {
                    hit.triangle = triangle;
                    hit.uvw = uvw;
                    hit.normal = normal;
                    hit.t = tMin;
                }
            }
        }
        const vec3 aabbHitT = (mix(node.aabbMax, node.aabbMin, corner) - ray.pos) * invDir;
        tMin = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
        if (expectEXT(hit.t <= tMin, false)) {
            break;
        }
        const ivec2 indices = mix(ivec2(0), ivec2(1, 2), equal(aabbHitT.yz, vec2(tMin)));
        const int ropeDirection = max(indices.x, indices.y);
        nodeIndex = corner[ropeDirection] ? node.leftRope[ropeDirection] : node.rightRope[ropeDirection];
    } while (expectEXT(nodeIndex != 0u, true));
}

layout(push_constant, scalar) uniform PushConstants
{
    vec4 clearColor;
    float tNear;
    vec3 pos;
    uint nodeIndex;
    Frustum frustum;
    vec2 viewportSize;
    float wireframeThickness;
    vec4 errorColor;
};

mat3 getViewMatrixX4()
{
    const vec3 topRight = frustum.rt - frustum.lb;
    const vec3 topLeft = frustum.lt - frustum.rb;
    // times 4 orts of camera space: right, up and forward
    return mat3(
        topRight - topLeft,
        topLeft + topRight,
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

void getAnalyticalBaryDeriv(const in Triangle triangle, const in vec3 p, const in vec3 uvw, out mat3x2 dduvw)
{
    const mat3 viewMatrixX4 = getViewMatrixX4();
    const vec3 a = projectToClip(triangle.a, viewMatrixX4);
    const vec3 b = projectToClip(triangle.b, viewMatrixX4);
    const vec3 c = projectToClip(triangle.c, viewMatrixX4);
    // unnormalized bary coords' derivs
    const vec2 ga = vec2(b.y - c.y, c.x - b.x);
    const vec2 gb = vec2(c.y - a.y, a.x - c.x);
    const vec2 gc = vec2(a.y - b.y, b.x - a.x);
    // clip-space signed area of triangle
    const float invArea = 1.0f / dot(ga, b.xy - a.xy);
    // scale
    const float pw = 4.0 * dot(p, viewMatrixX4[2]) * invArea;
    // persp-corrected gradients
    dduvw = pw * mat3x2(ga * a.z, gb * b.z, gc * c.z);
    dduvw = dduvw - outerProduct(dduvw[0] + dduvw[1] + dduvw[2], uvw);
}

void getBaryDeriv(const in vec3 rayDir, const in Hit hit, const in vec2 invViewportSize, out vec3 ddx, out vec3 ddy)
{
    // there is no sense in branching for calculation of all combinations
    // of these two conditions separately, because all these would be executed together in the same subgroup eventually
    if ((hit.triangle == subgroupQuadSwapHorizontal(hit.triangle)) && (hit.triangle == subgroupQuadSwapVertical(hit.triangle))) {
        ddx = dFdx(hit.uvw);
        ddy = dFdy(hit.uvw);
    } else {  // there are no counterparts in quad to correctly calculate perspective correct derivatives as differences using dFdx/dFdy
        mat3x2 dduvw;
        getAnalyticalBaryDeriv(getTriangle(hit.triangle), hit.t * rayDir, hit.uvw, dduvw);
        ddx = transpose(dduvw)[0] * invViewportSize.y;
        ddy = transpose(dduvw)[1] * invViewportSize.x;
    }
}

void main() [[maximally_reconverges]]
{
    const uvec2 imageExtent = uvec2(imageSize(target));
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
    // equiangularly
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
    traceRay(nodeIndex, ray, hit, tNear);
    vec4 color;
    if (hit.triangle != ~0u) {
        if (distance(hit.uvw, vec3(1.0f / 3.0f)) < 0.1f) {
            if (0.0f < wireframeThickness) {
                switch (hit.triangle % 3) {
                case 0 : {
                    color.rgb = vec3(0.0f, 0.0f, 1.0f);
                    break;
                }
                case 1 : {
                    color.rgb = vec3(0.0f, 1.0f, 0.0f);
                    break;
                }
                case 2 : {
                    color.rgb = vec3(1.0f, 0.0f, 0.0f);
                    break;
                }
                }
            } else {
                color.rgb = vec3(1.0f);
            }
        } else {
            if (0.0f < wireframeThickness) {
                vec3 ddx;
                vec3 ddy;
                getBaryDeriv(ray.dir, hit, 1.0f / viewportSize, ddx, ddy);
                color.rgb = getWireframeIntensity(hit.uvw, ddx, ddy, wireframeThickness).sss;
            } else {
                color.rgb = hit.uvw;
            }
        }
        color.a = clearColor.a;
    } else {
        color = clearColor;
    }
    if (any(isnan(color))) {
        color = errorColor;
    }
    imageStore(target, ivec2(gl_GlobalInvocationID.xy), color);
}
