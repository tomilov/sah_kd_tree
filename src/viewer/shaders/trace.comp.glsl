#version 460 core

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable

layout(local_size_x_id = 0, local_size_y_id = 1) in;
layout(constant_id = 2) const float kEps = 1E-7f;

struct Nodes  // sizeof(Node) == 64
{
    vec3 aabbMin;
    vec3 aabbMax;
    uvec3 leftRope;
    uvec3 rightRope;
    int splitDimension;
    float splitPos;
    uint leftChild;
    uint rightChild;
    //uint parent;
};

struct Triangle
{
    vec3 a, b, c;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer TreeTriangles
{
    Triangle triangle[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer TreePolygons
{
    uint polygon[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer TreeNodes
{
    Nodes node[];
};

struct Frustum
{
    vec3 leftTop, rightTop, leftBottom, rightBottom;
};

layout(push_constant, scalar) uniform PushConstants
{
    uvec2 size;
    TreeTriangles treeTriangles;
    TreePolygons treePolygons;
    TreeNodes treeNodes;
    uint startNodeIndex;
    vec3 pos;
    Frustum frustum;
    vec4 clearColor;
};

layout(binding = 0, rgba8) uniform image2D image;

struct Ray
{
    vec3 src;
    vec3 dir;
};

struct Hit
{
    float dist;
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

bool rayIntersectsTriangle(const in Ray ray, inout Hit hit, const in Triangle triangle, const in float tNear, const in float tFar)
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
            const int splitDimension = treeNodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (ray.src[splitDimension] < treeNodes.node[nodeIndex].splitPos) {
                nodeIndex = treeNodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = treeNodes.node[nodeIndex].rightChild;
            }
        }
    }
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    vec3 aabbHitDist = (mix(treeNodes.node[nodeIndex].aabbMax, treeNodes.node[nodeIndex].aabbMin, corner) - ray.src) * invDir;
    float tMin = min(aabbHitDist.x, min(aabbHitDist.y, aabbHitDist.z));
    do {
        const float tNear = min(0.0f, tMin);
        for (;;) {
            const int splitDimension = treeNodes.node[nodeIndex].splitDimension;
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == ((treeNodes.node[nodeIndex].splitPos - ray.src[splitDimension]) * invDir[splitDimension] < tNear)) {
                nodeIndex = treeNodes.node[nodeIndex].leftChild;
            } else {
                nodeIndex = treeNodes.node[nodeIndex].rightChild;
            }
        }
        aabbHitDist = (mix(treeNodes.node[nodeIndex].aabbMax, treeNodes.node[nodeIndex].aabbMin, corner) - ray.src) * invDir;
        const float tMax = min(aabbHitDist.x, min(aabbHitDist.y, aabbHitDist.z));
        if (tMin > tMax) {
            break;
        }
        const uint polygonStart = treeNodes.node[nodeIndex].leftChild;
        const uint polygonEnd = polygonStart + treeNodes.node[nodeIndex].rightChild;
        for (uint p = polygonStart; p < polygonEnd; ++p) {
            const float tFar = min(hit.dist, tMax);
            Hit newHit;
            newHit.triangle = treePolygons.polygon[p];
            if (rayIntersectsTriangle(ray, newHit, treeTriangles.triangle[newHit.triangle], tNear, tFar)) {
                hit = newHit;
            }
        }
        if (hit.dist <= tMax) {
            return true;
        }
        tMin = tMax;
        const uvec3 indices = mix(uvec3(0), uvec3(0, 1, 2), equal(aabbHitDist, vec3(tMax)));
        const uint ropeDirection = max(indices.x, max(indices.y, indices.z));
        nodeIndex = corner[ropeDirection] ? treeNodes.node[nodeIndex].leftRope[ropeDirection] : treeNodes.node[nodeIndex].rightRope[ropeDirection];
    } while (nodeIndex != 0);
    return false;
}

void main()
{
    const uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    if ((size.x >= pixelCoords.x) || (size.y >= pixelCoords.y)) {
        return;
    }
    const vec2 loc = (pixelCoords + 0.5f) / (gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    Ray ray;
    ray.src = pos;
    ray.dir = mix(
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
    Hit hit;
    hit.dist = 0.0f;
    vec4 color;
    if (traceRay(startNodeIndex, ray, hit)) {
        color = vec4(1.0f - (hit.uv.x + hit.uv.y), hit.uv, 1.0f);
    } else {
        color = clearColor;
    }
    imageStore(image, ivec2(pixelCoords), color);
}
