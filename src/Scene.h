#pragma once

#include "Types.h"
#include <string>
#include <vector>

namespace oops
{

    std::vector<Particle> makeScene(int count, const std::string &kind, unsigned seed);
    std::vector<Particle> loadScene(const std::string &path);
    void saveScene(const std::string &path, const std::vector<Particle> &particles);
    void mortonSort(std::vector<Particle> &particles);
    void saveSnapshot(const std::string &path, const std::vector<Particle> &particles,
                      const std::vector<Vec3> &forces, const std::vector<Contact> &contacts);

    // custom scene shapes
    enum class Shape
    {
        Box,
        Sphere,
        Cylinder
    };
    struct Block
    {
        Shape shape = Shape::Box;
        Vec3 center{0.5f, 0.45f, 0.5f};
        Vec3 size{0.5f, 0.5f, 0.5f};
        float radius = 0.012f;
        float spread = 0.f;
        Vec3 velocity{0.f, 0.f, 0.f};
    };
    struct SceneDesc
    {
        std::vector<Block> blocks{Block{}};
        float boxWidth = 1.f;
        float stiffness = 8000.f;
        unsigned seed = 1;
    };
    struct Preset
    {
        const char *name;
        SceneDesc scene;
    };
    std::vector<Particle> makeScene(const SceneDesc &desc);
    int estimateCount(const Block &block);
    float stableDt(const std::vector<Particle> &particles, float stiffness);
    std::vector<Preset> scenePresets();

} // namespace oops
