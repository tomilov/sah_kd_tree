#version 460 core

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable

layout(constant_id = 0) const uint kTriangleCount = 1;
layout(constant_id = 1) const uint kPolygonCount = 1;
layout(constant_id = 2) const uint kNodeCount = 1;
layout(constant_id = 3) const float kEps = 1E-7f;

struct Triangle
{
    vec3 a, b, c;
};

struct ProjectionNode  // TODO: transpose and fuse w/ Node
{
    float min[kNodeCount], max[kNodeCount];
    uint leftRope[kNodeCount], rightRope[kNodeCount];
};

struct Polygon
{
    uint triangle[kPolygonCount];
};

struct Node  // TODO: transpose and fuse w/ ProjectionNode
{
    int splitDimension[kNodeCount];
    float splitPos[kNodeCount];
    uint leftChild[kNodeCount];
    uint rightChild[kNodeCount];
    uint parent[kNodeCount];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer Tree
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

bool rayIntersectsTriangle(const in Tree tree, const in Ray ray, inout Hit hit, const in Triangle triangle, const in float tNear, const in float tFar)
{
    // TODO: Watertight Ray/Triangle Intersection, Sven Woop, Carsten Benthin, Ingo Wald
    vec3 a = triangle.a;
    vec3 edge1 = triangle.b - a;
    vec3 edge2 = triangle.c - a;
    vec3 normal = cross(edge1, edge2);
    float invNormal = 1.0f / length(normal);
    normal *= invNormal;
    float denominator2 = dot(normal, ray.dir);
    if (denominator2 == 0.0f) {
        return false;
    }
    if ((tNear - kEps > hit.distance) || (hit.distance > tFar + kEps)) {
        return false;
    }
    float xx = dot(edge1, edge1);
    float yy = dot(edge2, edge2);
    float xy = dot(edge1, edge2);
    float denominator = xy * xy - xx * yy;
    if (denominator == 0.0f) {
        return false;
    }
    denominator = 1.0f / denominator;
    vec3 intersection = ray.src + ray.dir * hit.distance;
    vec3 e = intersection - a;
    float ex = dot(e, edge1);
    float ey = dot(e, edge2);
    hit.uv.x = (xy * ey - yy * ex) * denominator;
    if (hit.uv.x < -kEps * invNormal * sqrt(xx)) {
        return false;
    }
    hit.uv.y = (xy * ex - xx * ey) * denominator;
    if (hit.uv.y < -kEps * invNormal * sqrt(yy)) {
        return false;
    }
    if ((1.0f - (hit.uv.x + hit.uv.y)) < -kEps * invNormal * length(edge1 - edge2)) {
        return false;
    }
    return true;
}

bool traceRay(const in Tree tree, in uint nodeIndex, const in Ray ray, inout Hit hit)
{
    if (nodeIndex == 0) {
        for (;;) {
            const int splitDimension = tree.node.splitDimension[nodeIndex];
            if (splitDimension < 0) {
                break;
            }
            if (ray.src[splitDimension] < tree.node.splitPos[nodeIndex]) {
                nodeIndex = tree.node.leftChild[nodeIndex];
            } else {
                nodeIndex = tree.node.rightChild[nodeIndex];
            }
        }
    }
    // https://people.csail.mit.edu/amy/papers/box-jgt.pdf (An efficient and robust ray-box intersection algorithm)
    const vec3 invDir = 1.0f / clearZeroSign(ray.dir);
    const bvec3 corner = lessThan(invDir, vec3(0.0f));
    vec3 aabbMin = vec3(tree.x.min[nodeIndex], tree.y.min[nodeIndex], tree.z.min[nodeIndex]);  // TODO: SoA to AoS for that (ProjectionNode.xyz.min -> ProjectionNode.min.xyz)
    vec3 aabbMax = vec3(tree.x.max[nodeIndex], tree.y.max[nodeIndex], tree.z.max[nodeIndex]);  // TODO: SoA to AoS for that (ProjectionNode.xyz.max -> ProjectionNode.max.xyz)
    vec3 aabbHitDistances = (mix(aabbMin, aabbMax, corner) - ray.src) * invDir;
    float tMin = min(aabbHitDistances.x, min(aabbHitDistances.y, aabbHitDistances.z));
    do {
        const float tNear = min(0.0f, tMin);
        for (;;) {
            const int splitDimension = tree.node.splitDimension[nodeIndex];
            if (splitDimension < 0) {
                break;
            }
            if (corner[splitDimension] == ((tree.node.splitPos[nodeIndex] - ray.src[splitDimension]) * invDir[splitDimension] < tNear)) {
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
        const uint polygonStart = tree.node.leftChild[nodeIndex];
        const uint polygonEnd = polygonStart + tree.node.rightChild[nodeIndex];
        for (uint t = polygonStart; t < polygonEnd; ++t) {
            float tFar = min(hit.distance, tMax);
            Hit newHit;
            newHit.triangle = tree.polygon.triangle[t];
            if (rayIntersectsTriangle(tree, ray, newHit, tree.triangles[hit.triangle], tNear, tFar)) {
                hit = newHit;
            }
        }
        if (hit.distance <= tMax) {
            return true;
        }
        tMin = tMax;
        const uvec3 indices = mix(uvec3(0), uvec3(0, 1, 2), equal(aabbHitDistances, vec3(tMax)));
        const uint ropeDirection = max(indices.x, max(indices.y, indices.z));
        const uvec3 leftRopes = uvec3(tree.x.leftRope[nodeIndex], tree.y.leftRope[nodeIndex], tree.z.leftRope[nodeIndex]);
        const uvec3 rightRopes = uvec3(tree.x.rightRope[nodeIndex], tree.y.rightRope[nodeIndex], tree.z.rightRope[nodeIndex]);
        nodeIndex = mix(leftRopes[ropeDirection], rightRopes[ropeDirection], corner[ropeDirection]);
    } while (nodeIndex != 0);
    return false;
}

struct Frustum
{
    vec3 leftTop, rightTop, leftBottom, rightBottom;
};

layout(local_size_x_id = 4, local_size_y_id = 5) in;

layout(binding = 0, rgba8) uniform image2D image;

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
    hit.distance = 0.0f;
    vec4 color;
    if (traceRay(tree, nodeIndex, ray, hit)) {
        color = vec4(1.0f - hit.uv.x - hit.uv.y, hit.uv, 1.0f);
    } else {
        color = clearColor;
    }
    imageStore(image, ivec2(pixelCoords), color);
}
