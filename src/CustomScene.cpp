// Builds scenes out of shaped blocks of grains.
#include "Scene.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>

namespace oops
{

    namespace
    {
        constexpr float pi = 3.14159265358979323846f;
        constexpr size_t maxGrains = 300000;

        // spatial hash of grain centers to reject grains that would overlap
        class OverlapGrid
        {
        public:
            explicit OverlapGrid(float cellSize) : cell_(cellSize) {}

            void insert(const Particle &p)
            {
                cells_[key(coord(p.position.x), coord(p.position.y), coord(p.position.z))].push_back(p);
            }

            bool overlaps(Vec3 position, float radius) const
            {
                int cx = coord(position.x), cy = coord(position.y), cz = coord(position.z);
                for (int x = cx - 1; x <= cx + 1; ++x)
                    for (int y = cy - 1; y <= cy + 1; ++y)
                        for (int z = cz - 1; z <= cz + 1; ++z)
                        {
                            auto found = cells_.find(key(x, y, z));
                            if (found == cells_.end())
                                continue;
                            for (const Particle &other : found->second)
                            {
                                Vec3 d = other.position - position;
                                float reach = other.radius + radius;
                                if (dot(d, d) < reach * reach)
                                    return true;
                            }
                        }
                return false;
            }

        private:
            int coord(float value) const { return int(std::floor(value / cell_)); }
            static int64_t key(int x, int y, int z)
            {
                return (int64_t(x & 0x1FFFFF) << 42) | (int64_t(y & 0x1FFFFF) << 21) | int64_t(z & 0x1FFFFF);
            }
            float cell_;
            std::unordered_map<int64_t, std::vector<Particle>> cells_;
        };

        bool insideShape(const Block &block, Vec3 p)
        {
            Vec3 half = 0.5f * block.size;
            Vec3 d = p - block.center;
            float nx = d.x / std::max(half.x, 1e-6f);
            float ny = d.y / std::max(half.y, 1e-6f);
            float nz = d.z / std::max(half.z, 1e-6f);
            switch (block.shape)
            {
            case Shape::Sphere:
                return nx * nx + ny * ny + nz * nz <= 1.f;
            case Shape::Cylinder:
                return nx * nx + nz * nz <= 1.f;
            default:
                return true;
            }
        }

        float latticeSpacing(const Block &block) { return 2.f * block.radius * 1.04f; }

        int latticeCount(float extent, float spacing)
        {
            return std::max(1, int(std::floor(extent / spacing)));
        }

        float grainMass(float radius)
        {
            return 2500.f * (4.f / 3.f) * pi * radius * radius * radius;
        }
    } // namespace (not oops)

    int estimateCount(const Block &block)
    {
        float spacing = latticeSpacing(block);
        double lattice = double(latticeCount(block.size.x, spacing)) * latticeCount(block.size.y, spacing) * latticeCount(block.size.z, spacing);
        double fill = block.shape == Shape::Sphere ? pi / 6.f : block.shape == Shape::Cylinder ? pi / 4.f
                                                                                               : 1.f;
        return std::max(1, int(lattice * fill));
    }

    std::vector<Particle> makeScene(const SceneDesc &desc)
    {
        float largest = 0.f;
        for (const Block &block : desc.blocks)
            largest = std::max(largest, block.radius);

        std::vector<Particle> particles;
        if (!(largest > 0.f))
            return particles;
        OverlapGrid grid(2.f * largest);
        std::mt19937 random(desc.seed);
        std::uniform_real_distribution<float> uniform(0.f, 1.f);

        for (const Block &block : desc.blocks)
        {
            float spacing = latticeSpacing(block);
            int nx = latticeCount(block.size.x, spacing);
            int ny = latticeCount(block.size.y, spacing);
            int nz = latticeCount(block.size.z, spacing);
            Vec3 start = block.center - 0.5f * spacing * Vec3(float(nx - 1), float(ny - 1), float(nz - 1));

            for (int y = 0; y < ny; ++y)
                for (int z = 0; z < nz; ++z)
                    for (int x = 0; x < nx; ++x)
                    {
                        if (particles.size() >= maxGrains)
                            return particles;
                        Vec3 jitter(uniform(random) - 0.5f, uniform(random) - 0.5f, uniform(random) - 0.5f);
                        Vec3 position = start + spacing * Vec3(float(x), float(y), float(z)) + 0.02f * spacing * jitter;
                        float radius = block.radius * (1.f - block.spread * uniform(random));
                        if (!insideShape(block, position))
                            continue;

                        bool outside = false;
                        for (int axis = 0; axis < 3; ++axis)
                            outside |= position[axis] < radius || position[axis] > desc.boxWidth - radius;
                        if (outside || grid.overlaps(position, radius))
                            continue;

                        Particle p;
                        p.position = position;
                        p.radius = radius;
                        p.mass = grainMass(radius);
                        p.velocity = block.velocity;
                        particles.push_back(p);
                        grid.insert(p);
                    }
        }
        return particles;
    }

    float stableDt(const std::vector<Particle> &particles, float stiffness)
    {
        float lightest = 1e30f;
        for (const Particle &p : particles)
            lightest = std::min(lightest, p.mass);
        float collisionTime = pi * std::sqrt(0.5f * lightest / stiffness);
        return collisionTime / 15.f; // resolve each collision with about 15 steps
    }

    std::vector<Preset> scenePresets()
    {
        auto block = [](Shape shape, Vec3 center, Vec3 size, float radius)
        {
            Block b;
            b.shape = shape;
            b.center = center;
            b.size = size;
            b.radius = radius;
            return b;
        };
        auto scene = [](std::vector<Block> blocks, float stiffness = 8000.f)
        {
            SceneDesc s;
            s.blocks = std::move(blocks);
            s.stiffness = stiffness;
            return s;
        };

        Block mixed = block(Shape::Box, Vec3(0.5f, 0.45f, 0.5f), Vec3(0.55f), 0.012f);
        mixed.spread = 0.3f;

        Block column = block(Shape::Cylinder, Vec3(0.5f, 0.41f, 0.5f), Vec3(0.3f, 0.8f, 0.3f), 0.01f);
        Block dam = block(Shape::Box, Vec3(0.17f, 0.33f, 0.5f), Vec3(0.32f, 0.64f, 0.96f), 0.011f);

        Block left = block(Shape::Box, Vec3(0.25f, 0.3f, 0.5f), Vec3(0.28f), 0.01f);
        Block right = left;
        right.center.x = 0.75f;
        left.velocity = Vec3(3.f, 0.f, 0.f);
        right.velocity = Vec3(-3.f, 0.f, 0.f);

        Block bed = block(Shape::Box, Vec3(0.5f, 0.08f, 0.5f), Vec3(0.98f, 0.15f, 0.98f), 0.01f);
        Block ball = block(Shape::Sphere, Vec3(0.5f, 0.7f, 0.5f), Vec3(0.28f), 0.012f);
        ball.velocity = Vec3(0.f, -4.f, 0.f);

        return {
            {"Cube drop", scene({mixed})},
            {"Column collapse", scene({column})},
            {"Dam break", scene({dam})},
            {"Two blocks", scene({left, right}, 40000.f)},
            {"Meteor", scene({bed, ball}, 40000.f)},
        };
    }

}
