//#pragma once

float getWireFrameIntensity(vec3 baryCoord, vec3 dBaryCoordX, vec3 dBaryCoordY, float thickness)
{
    vec3 dBaryCoord = sqrt(dBaryCoordX * dBaryCoordX + dBaryCoordY * dBaryCoordY);
    vec3 dThickness = dBaryCoord * thickness;
    vec3 remap = step(dThickness, baryCoord);
    return min(min(remap.x, remap.y), remap.z);
}
