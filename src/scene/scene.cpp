#include <scene/scene.hpp>

namespace scene
{

void Scene::resize(size_t newTraingleCount)
{
    triangleCount = newTraingleCount;
    triangles = std::make_unique<Triangle[]>(newTraingleCount);
}

}
