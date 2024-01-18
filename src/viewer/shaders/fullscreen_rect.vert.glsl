#version 460 core

#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_debug_printf : enable

//layout(location = 0) in vec2 vertices;
layout(location = 0) out vec2 uv;

out gl_PerVertex { vec4 gl_Position; };

layout(push_constant, scalar) uniform PushConstants
{
    layout(row_major) mat4x3 viewTransform;
} pushConstants;

void main()
{
    //gl_Position = vec4(mat3(pushConstants.viewTransform) * vec3(vertices, 0.0f), 1.0f);
    switch (gl_VertexIndex) {
    case 0: {
        uv = vec2(-1.0f, -1.0f);
        break;
    }
    case 1: {
        uv = vec2(-1.0f, 1.0f);
        break;
    }
    case 2: {
        uv = vec2(1.0f, -1.0f);
        break;
    }
    case 3: {  // TODO: make n of vertices 3 and remove me
        uv = vec2(1.0f, 1.0f);
        break;
    }
    }
    gl_Position = vec4(vec3(uv, 0.0f) * mat3(pushConstants.viewTransform), 1.0f);
    debugPrintfEXT("%i\n", gl_VertexIndex);
}

