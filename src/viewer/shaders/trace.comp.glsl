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
    vec3 leftTop;
    vec3 rightTop;
    vec3 leftBottom;
    vec3 rightBottom;
};

layout(set = 0, binding = 0, scalar) uniform UniformBuffer
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
    vec3 src;
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

// https://iquilezles.org/articles/intersectors/
bool intersectSphere(const in Ray ray, const in vec3 center, const in float radius)
{
    const vec3 oc = center - ray.src;
    const float l = dot(ray.dir, oc);
    if (l < 0.0f) {
        return false;
    }
    const vec3 ll = ray.dir * l;
    return radius * radius > dot(oc, oc) - dot(ll, ll);
}

bool rayTriangleIntersect(const in Ray ray, inout Hit hit, const in Triangle triangle, const in float tNear, const in float tFar)
{
    // TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
    const vec3 v1v0 = triangle.b - triangle.a;
    const vec3 v2v0 = triangle.c - triangle.a;
    const vec3 rov0 = ray.src - triangle.a;
    const vec3 n = cross(v1v0, v2v0);
    const float d = 1.0f / dot(ray.dir, n);
    hit.t = d * dot(-n, rov0);
    if ((hit.t < tNear) || (tFar < hit.t)) {
        return false;
    }
    const vec3 q = cross(rov0, ray.dir);
    hit.uv.x = d * dot(-q, v2v0);
    hit.uv.y = d * dot(q, v1v0);
    if ((hit.uv.x < 0.0f) || (hit.uv.y < 0.0f) || (hit.uv.x + hit.uv.y > 1.0f)) {
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
    vec3 aabbHitT = (mix(nodes.node[nodeIndex].aabbMin, nodes.node[nodeIndex].aabbMax, corner) - ray.src) * invDir;
    float tMin = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
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
        aabbHitT = (mix(nodes.node[nodeIndex].aabbMax, nodes.node[nodeIndex].aabbMin, corner) - ray.src) * invDir;
        const float tMax = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
        if (tMin > tMax) {
            break;
        }
        const uint polygonStart = nodes.node[nodeIndex].leftChild;
        const uint polygonEnd = polygonStart + nodes.node[nodeIndex].rightChild;
        for (uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
            const float tFar = min(hit.t, tMax);
            Hit closerHit;
            closerHit.triangle = polygons.triangle[polygon];
            if (rayTriangleIntersect(ray, closerHit, triangles.triangle[closerHit.triangle], tNear, tFar)) {
                if (closerHit.t < hit.t) {
                    hit = closerHit;
                }
            }
        }
        if (hit.t <= tMax) {
            return true;
        }
        tMin = tMax;
        const uvec3 indices = mix(uvec3(0), uvec3(0, 1, 2), equal(aabbHitT, vec3(tMax)));
        const uint ropeDirection = max(indices.x, max(indices.y, indices.z));
        nodeIndex = corner[ropeDirection] ? nodes.node[nodeIndex].leftRope[ropeDirection] : nodes.node[nodeIndex].rightRope[ropeDirection];
    } while (nodeIndex != 0);
    return false;
}

layout(push_constant, scalar) uniform PushConstants
{
    vec4 clearColor;
    vec3 pos;
    uint nodeIndex;
    Frustum frustum;
};

void main()
{
    const uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    const ivec2 imageSize = imageSize(target);
    if ((imageSize.x <= pixelCoords.x) || (imageSize.y <= pixelCoords.y)) {
        return;
    }
    if ((imageSize.x == pixelCoords.x + 1) || (imageSize.y == pixelCoords.y + 1) || (pixelCoords.x == 0) || (pixelCoords.y == 0)) {
        imageStore(target, ivec2(pixelCoords), vec4(0.0f, 1.0f, 0.0f, 1.0f));
        return;
    }
    const vec2 loc = (pixelCoords + 0.5f) / imageSize;
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
    hit.t = 0.0f;
    vec4 color;
#if 0
    if (traceRay(nodeIndex, ray, hit)) {
        color = vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv, 1.0f / (1.0f + hit.t));
#elif 1
    Triangle triangle;
    triangle.a = vec3(0.0f, 0.0f, 0.0f);
    triangle.b = vec3(1.0f, 1.0f, 0.0f);
    triangle.c = vec3(-1.0f, 1.0f, 0.0f);
    if (rayTriangleIntersect(ray, hit, triangle, 0.0f, 2.0f) && ((nodeCount != 0) || (nodeCount == 0))) {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
#else
    if (intersectSphere(ray, vec3(0.0f), 1.0f) && ((nodeCount != 0) || (nodeCount == 0))) {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
#endif
    } else {
        color = clearColor;
    }
    imageStore(target, ivec2(pixelCoords), color);
}
