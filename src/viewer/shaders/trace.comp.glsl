#version 460 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#define sizeof(Type) (uint64_t(Type(uint64_t(0))+1))

layout(local_size_x_id = 0, local_size_y_id = 1) in;
layout(constant_id = 2) const float kEps = 1E-7f;

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
    vec3 leftTop, rightTop, leftBottom, rightBottom;
};

layout(set = 0, binding = 0, scalar) uniform UniformBuffer
{
    uint treeDepthMax;
    Triangles triangles;
    Polygons polygons;
    Nodes nodes;
    NodeParents nodeParents;
};

layout(set = 1, binding = 0, rgba8) uniform image2D target;

struct Ray
{
    vec3 src;
    vec3 dir;
};

struct Hit
{
    uint triangle;
    float dist;
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

bool rayTriangleIntersect(const in Ray ray, inout Hit hit, const in Triangle triangle, const in float tNear, const in float tFar)
{
    // TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
    const vec3 edge1 = triangle.b - triangle.a;
    const vec3 edge2 = triangle.c - triangle.a;
    vec3 normal = cross(edge1, edge2);
    const float denominator2 = dot(normal, ray.dir);
    if (abs(denominator2) < kEps) {
        return false;
    }
    float invNormal = 1.0f / length(normal);
    normal *= invNormal;
    hit.dist = dot(normal, triangle.a - ray.src);
    if ((tNear - kEps >= hit.dist) || (hit.dist >= tFar + kEps)) {
        return false;
    }
    const float xx = dot(edge1, edge1);
    const float yy = dot(edge2, edge2);
    const float xy = dot(edge1, edge2);
    float denominator = xy * xy - xx * yy;
    if (abs(denominator) <= kEps) {
        return false;
    }
    denominator = 1.0f / denominator;
    const vec3 intersection = ray.src + ray.dir * hit.dist;
    const vec3 e = intersection - triangle.a;
    const float ex = dot(e, edge1);
    const float ey = dot(e, edge2);
    invNormal *= -kEps;
    hit.uv.x = (xy * ey - yy * ex) * denominator;
    if (hit.uv.x < invNormal * sqrt(xx)) {
        return false;
    }
    hit.uv.y = (xy * ex - xx * ey) * denominator;
    if (hit.uv.y < invNormal * sqrt(yy)) {
        return false;
    }
    if (1.0f - (hit.uv.x + hit.uv.y) < invNormal * length(edge1 - edge2)) {
        return false;
    }
    return true;
}

bool traceRay(in uint nodeIndex, const in Ray ray, inout Hit hit)
{
    if (nodeIndex == 0) {
        for (;;) {
            const int splitDimension = nodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (ray.src[splitDimension] < nodes.node[nodeIndex].splitPos) {
                nodeIndex = nodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = nodes.node[nodeIndex].rightChild;
            }
        }
    }
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    vec3 aabbHitDist = (mix(nodes.node[nodeIndex].aabbMin, nodes.node[nodeIndex].aabbMax, corner) - ray.src) * invDir;
    float tMin = min(aabbHitDist.x, min(aabbHitDist.y, aabbHitDist.z));
    do {
        const float tNear = min(0.0f, tMin);  // adjust for the case if we are not outside
        for (;;) {
            const int splitDimension = nodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == ((nodes.node[nodeIndex].splitPos - ray.src[splitDimension]) * invDir[splitDimension] < tNear)) {
                nodeIndex = nodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = nodes.node[nodeIndex].rightChild;
            }
        }
        aabbHitDist = (mix(nodes.node[nodeIndex].aabbMax, nodes.node[nodeIndex].aabbMin, corner) - ray.src) * invDir;
        const float tMax = min(aabbHitDist.x, min(aabbHitDist.y, aabbHitDist.z));
        if (tMin > tMax) {
            break;
        }
        const uint polygonStart = nodes.node[nodeIndex].leftChild;
        const uint polygonEnd = polygonStart + nodes.node[nodeIndex].rightChild;
        for (uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
            const float tFar = min(hit.dist, tMax);
            Hit closerHit;
            closerHit.triangle = polygons.triangle[polygon];
            if (rayTriangleIntersect(ray, closerHit, triangles.triangle[closerHit.triangle], tNear, tFar)) {
                if (closerHit.dist < hit.dist) {
                    hit = closerHit;
                }
            }
        }
        if (hit.dist <= tMax) {
            return true;
        }
        tMin = tMax;
        const uvec3 indices = mix(uvec3(0), uvec3(0, 1, 2), equal(aabbHitDist, vec3(tMax)));
        const uint ropeDirection = max(indices.x, max(indices.y, indices.z));
        nodeIndex = corner[ropeDirection] ? nodes.node[nodeIndex].leftRope[ropeDirection] : nodes.node[nodeIndex].rightRope[ropeDirection];
    } while (nodeIndex != 0);
    return false;
}

layout(push_constant, scalar) uniform PushConstants
{
    vec4 clearColor;
    vec3 pos;
    Frustum frustum;
    uint nodeIndex;
};

void main()
{
    const uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    const ivec2 imageSize = imageSize(target);
    if ((imageSize.x <= pixelCoords.x) || (imageSize.y <= pixelCoords.y)) {
        return;
    }
    const vec2 loc = (pixelCoords + 0.5f) / (gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    const vec3 dir = mix(
        mix(
            frustum.leftBottom,
            frustum.rightBottom,
            loc.x
        ),
        mix(
            frustum.leftTop,
            frustum.rightTop,
            loc.x
        ),
        loc.y
    );
    Ray ray;
    ray.src = pos;
    ray.dir = normalize(dir);
    Hit hit;
    hit.dist = 0.0f;
    vec4 color;
    if (traceRay(nodeIndex, ray, hit)) {
        color = vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv, 1.0f / (1.0f + hit.dist));
    } else {
        color = clearColor;
    }
    imageStore(target, ivec2(pixelCoords), color);
}
