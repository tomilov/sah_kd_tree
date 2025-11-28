//#pragma once

float getWireframeIntensity(vec3 uvw, vec3 ddx, vec3 ddy, float thickness)
{
    vec3 duvw = sqrt(ddx * ddx + ddy * ddy);
    vec3 dThickness = duvw * thickness;
    vec3 remap = step(dThickness, uvw);
    return min(min(remap.x, remap.y), remap.z);
}
