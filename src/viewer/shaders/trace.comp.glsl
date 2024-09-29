#version 460 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

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
    return !((hit.uv.x < 0.0f) || (hit.uv.y < 0.0f) || (hit.uv.x + hit.uv.y > 1.0f));
}

bool traceRay(in uint nodeIndex, const in Ray ray, inout Hit hit, const in bool debug)
{
    int i = 0;
    if (nodeIndex == 0u) {
        for (;;) {
            if (++i == 200) {
                //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
                return false;
            }
            const int splitDimension = nodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (ray.src[splitDimension] < nodes.node[nodeIndex].splitPos) {
                nodeIndex = nodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = nodes.node[nodeIndex].rightChild;
            }
            //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
        }
    }
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    vec3 aabbHitT = (mix(nodes.node[nodeIndex].aabbMin, nodes.node[nodeIndex].aabbMax, corner) - ray.src) * invDir;
    float tMin = max(aabbHitT.x, max(aabbHitT.y, aabbHitT.z));
    if (debug) {
        //vec3 a = nodes.node[nodeIndex].aabbMin;
        //vec3 b = nodes.node[nodeIndex].aabbMax;
        //debugPrintfEXT("%f %f %f %f %f %f\n", a.x, a.y, a.z, b.x, b.y, b.z);
        //vec3 aaa = aabbHitT;
        //debugPrintfEXT("%f %f %f\n", aaa.x, aaa.y, aaa.z);
    }
    //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
    do {
        const float tNear = max(0.0f, tMin);  // adjust for the case if we are not outside
        for (;;) {
            if (++i == 200) {
                //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
                return false;
            }
            const int splitDimension = nodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == ((nodes.node[nodeIndex].splitPos - ray.src[splitDimension]) * invDir[splitDimension] < tNear)) {
                nodeIndex = nodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = nodes.node[nodeIndex].rightChild;
            }
            //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
        }
        //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
        aabbHitT = (mix(nodes.node[nodeIndex].aabbMax, nodes.node[nodeIndex].aabbMin, corner) - ray.src) * invDir;
        const float tMax = min(aabbHitT.x, min(aabbHitT.y, aabbHitT.z));
        //hit.uv = vec2(aabbHitT.y, aabbHitT.z) / max(aabbHitT.x, max(aabbHitT.y, aabbHitT.z));
        //if (debug) {
        //    debugPrintfEXT("%f %f\n", tMin, tMax);
        //}
        //debugPrintfEXT("%i %u %u %u\n", __LINE__, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, nodes.node[nodeIndex].rightChild);
        if (tMin > tMax) {
            //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
            break;
        }
        //return true;
        //if (debug) {
        //    debugPrintfEXT("START\n");
        //}
        const uint polygonStart = nodes.node[nodeIndex].leftChild;
        const uint polygonEnd = polygonStart + nodes.node[nodeIndex].rightChild;
        for (uint polygon = polygonStart; polygon < polygonEnd; ++polygon) {
            if (++i == 200) {
                //debugPrintfEXT("%i %u %u %u\n", __LINE__, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, nodes.node[nodeIndex].rightChild);
                return true;
            }
            const float tFar = min(hit.t, tMax);
            Hit closerHit;
            closerHit.triangle = polygons.triangle[polygon];
            if (rayTriangleIntersect(ray, closerHit, triangles.triangle[closerHit.triangle], tNear, tFar)) {
                //if (debug) {
                //    debugPrintfEXT("%i %u %f %f %f %f\n", __LINE__, polygon, tNear, tFar, closerHit.t, hit.t);
                //}
                if (closerHit.t < hit.t) {
                    hit = closerHit;
                }
            } else {
                //if (debug) {
                //    debugPrintfEXT("%i %u\n", __LINE__, polygon);
                //}
            }
        }
        if (hit.t <= tMax) {
            //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
            return true;
        }
        tMin = tMax;
        const ivec3 indices = mix(ivec3(0), ivec3(0, 1, 2), equal(aabbHitT, vec3(tMax)));
        const int ropeDirection = max(indices.x, max(indices.y, indices.z));
        nodeIndex = corner[ropeDirection] ? nodes.node[nodeIndex].leftRope[ropeDirection] : nodes.node[nodeIndex].rightRope[ropeDirection];
        //if (debug) {
        //    debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
        //}
        debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
    } while (nodeIndex != 0u);
    //debugPrintfEXT("%i %u\n", __LINE__, nodeIndex);
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
    hit.t = +1.0f / +0.0f;
    vec4 color;
#if 1
    bool debug = (2u * pixelCoords == uvec2(imageSize));
    if (!debug) {
        //return;
    }
    if (traceRay(nodeIndex, ray, hit, debug)) {
        color = vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv.yx, 1.0f);
#elif 0
    Triangle triangle;
    triangle.a = vec3(0.0f, 0.0f, 0.0f);
    triangle.b = vec3(1.0f, 1.0f, 0.0f);
    triangle.c = vec3(-1.0f, 1.0f, 0.0f);
    if (rayTriangleIntersect(ray, hit, triangle, 0.0f, 20.0f) && ((nodeCount != 0u) || (nodeCount == 0u))) {
        color = vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv.yx, 1.0f);
#else
    if (intersectSphere(ray, vec3(0.0f), 1.0f) && ((nodeCount != 0u) || (nodeCount == 0u))) {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
#endif
    } else {
        color = clearColor;
    }
    imageStore(target, ivec2(pixelCoords), color);
}
