#include "Scene.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace oops {

std::vector<Particle> makeScene(int count, const std::string& kind, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);
    int side = int(std::ceil(std::cbrt(double(count))));
    float cell = 0.7f / side;
    std::vector<Particle> particles(count);
    for (int i = 0; i < count; ++i) {
        Vec3 grid(float(i % side), float((i / side) % side), float(i / (side * side)));
        Vec3 jitter(uniform(random) - 0.5f, uniform(random) - 0.5f, uniform(random) - 0.5f);
        auto& p = particles[i];
        p.position = Vec3(0.15f, 0.25f, 0.15f) + cell * (grid + Vec3(0.5f) + 0.15f * jitter);
        p.radius = cell * (kind == "mixed" ? 0.15f + 0.45f * uniform(random) : 0.4f);
        if (kind == "clustered") p.position = Vec3(0.5f) + 0.5f * (p.position - Vec3(0.5f));
        p.mass = 2500.f * (4.f / 3.f) * 3.14159265358979323846f
            * p.radius * p.radius * p.radius;
    }
    return particles;
}

std::vector<Particle> loadScene(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open scene: " + path);
    std::vector<Particle> particles;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '#') continue;
        Particle p;
        std::istringstream row(line);
        if (!(row >> p.position.x >> p.position.y >> p.position.z >> p.radius >> p.mass)
            || p.radius <= 0.f || p.mass <= 0.f)
            throw std::runtime_error("Expected: x y z radius mass [vx vy vz] in " + path);
        if (row >> p.velocity.x) {
            if (!(row >> p.velocity.y >> p.velocity.z))
                throw std::runtime_error("Velocity needs three components in " + path);
        }
        particles.push_back(p);
    }
    return particles;
}

void saveScene(const std::string& path, const std::vector<Particle>& particles) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not write scene: " + path);
    file << std::setprecision(9);
    for (const auto& p : particles)
        file << p.position.x << ' ' << p.position.y << ' ' << p.position.z << ' '
             << p.radius << ' ' << p.mass << ' ' << p.velocity.x << ' '
             << p.velocity.y << ' ' << p.velocity.z << '\n';
}

namespace {
uint32_t expandBits(uint32_t value) {
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    return (value * 0x00000005u) & 0x49249249u;
}
uint32_t mortonKey(Vec3 position) {
    uint32_t axis[3];
    for (int i = 0; i < 3; ++i)
        axis[i] = expandBits(uint32_t(std::clamp(position[i] * 1024.f, 0.f, 1023.f)));
    return 4 * axis[0] + 2 * axis[1] + axis[2];
}
}

void mortonSort(std::vector<Particle>& particles) {
    std::vector<size_t> order(particles.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        uint32_t ka = mortonKey(particles[a].position), kb = mortonKey(particles[b].position);
        return ka != kb ? ka > kb : a > b;
    });
    auto original = particles;
    for (size_t i = 0; i < order.size(); ++i) particles[i] = original[order[i]];
}

void saveSnapshot(const std::string& path, const std::vector<Particle>& particles,
                  const std::vector<Vec3>& forces, const std::vector<Contact>& contacts) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not write snapshot: " + path);
    file << "id,x,y,z,radius,mass,vx,vy,vz,fx,fy,fz\n" << std::setprecision(9);
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        const auto& f = forces[i];
        file << i << ',' << p.position.x << ',' << p.position.y << ',' << p.position.z
             << ',' << p.radius << ',' << p.mass << ',' << p.velocity.x << ','
             << p.velocity.y << ',' << p.velocity.z << ',' << f.x << ',' << f.y << ',' << f.z << '\n';
    }
    std::ofstream pairs(path + ".pairs");
    for (auto c : contacts) pairs << c.a << ' ' << c.b << '\n';
}

} // namespace oops
