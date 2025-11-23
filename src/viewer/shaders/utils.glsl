float getWireFrameIntensity(in vec3 baryCoord, in float thickness)
{
    vec3 dBaryCoordX = dFdxFine(baryCoord);
    vec3 dBaryCoordY = dFdyFine(baryCoord);
    vec3 dBaryCoord  = sqrt(dBaryCoordX * dBaryCoordX + dBaryCoordY * dBaryCoordY);

    vec3 dThickness = dBaryCoord * thickness;

    vec3 remap = step(dThickness, baryCoord);
    float closestEdge = min(min(remap.x, remap.y), remap.z);

    return closestEdge;
}
