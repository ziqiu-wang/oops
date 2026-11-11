// AI-generated test suite for checking correctness

#include "Simulation.h"
#include "Reference.h"
#include "Scene.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

using namespace oops;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void expectNear(Vec3 a, Vec3 b, const std::string& message, float absolute = 0.001f) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!(std::abs(a[axis] - b[axis]) <= absolute + 0.00005f * std::abs(b[axis])))
            throw std::runtime_error(message + ": got " + std::to_string(a[axis])
                                     + ", expected " + std::to_string(b[axis]));
    }
}
Particle sphere(Vec3 center, float radius, float mass = 1.f, Vec3 velocity = Vec3(0.f)) {
    return {center, radius, velocity, mass};
}

void cpuTests() {
    require(referenceContacts({}).empty(), "empty reference scene");
    auto a = sphere(Vec3(0.f), 0.5f);
    auto b = sphere(Vec3(0.75f, 0.f, 0.f), 0.5f);
    Settings settings;
    settings.gravity = Vec3(0.f);
    settings.restitution = 1.f;
    auto forces = referenceForces({a, b}, settings);
    expectNear(forces[0], Vec3(-2000.f, 0.f, 0.f), "analytical spring");
    expectNear(forces[0] + forces[1], Vec3(0.f), "action/reaction");

    settings.gravity = Vec3(0.f, -10.f, 0.f);
    settings.dt = 0.001f;
    settings.boxWidth = 0.f;
    ReferenceSimulation falling({sphere(Vec3(0.f), 0.1f)}, settings);
    falling.step();
    expectNear(falling.particles[0].position, Vec3(0.f), "upstream zero-force first step", 1.e-8f);
    falling.step();
    expectNear(falling.particles[0].velocity, Vec3(0.f, -0.005f, 0.f), "upstream velocity", 1.e-8f);
    expectNear(falling.particles[0].position, Vec3(0.f, -0.00001f, 0.f), "upstream position", 1.e-8f);

    settings.gravity = Vec3(0.f);
    settings.boxWidth = 1.f;
    settings.dt = 0.01f;
    ReferenceSimulation wall({sphere(Vec3(0.051f, 0.5f, 0.5f), 0.05f, 1.f, Vec3(-1.f, 0.f, 0.f))}, settings);
    wall.step();
    expectNear(wall.particles[0].position, Vec3(0.05f, 0.5f, 0.5f), "wall clamping", 1.e-7f);
    expectNear(wall.particles[0].velocity, Vec3(0.1f, 0.f, 0.f), "wall restitution", 1.e-7f);
    std::cout << "PASS CPU analytical force, integration, and walls\n";
}

void checkScene(const std::string& name, const std::vector<Particle>& particles,
                bool strict = false) {
    Settings settings;
    settings.gravity = Vec3(0.f);
    settings.strictUpstream = strict;
    Simulation gpu(particles, settings, Response::Contacts, 1); // Exercises overflow/retry.
    auto expectedPairs = referenceContacts(particles, strict);
    auto pairs = gpu.contacts();
    require(pairs == expectedPairs, name + ": exact contact set");
    require(std::adjacent_find(pairs.begin(), pairs.end()) == pairs.end(), name + ": duplicates");
    auto expectedForces = referenceForces(particles, settings);
    for (auto mode : {Response::Contacts, Response::Fused}) {
        gpu.response = mode;
        gpu.computeForces();
        auto actual = gpu.forces();
        Vec3 total(0.f);
        for (size_t i = 0; i < particles.size(); ++i) {
            expectNear(actual[i], expectedForces[i], name + ": force");
            total += actual[i];
        }
        expectNear(total, Vec3(0.f), name + ": total contact force", 0.02f);
    }
    auto changed = particles;
    for (size_t i = 0; i < changed.size(); ++i) {
        changed[i].position += Vec3(float(i) * 0.1f, 0.f, 0.f);
        changed[i].radius *= 1.5f;
    }
    gpu.upload(changed);
    require(gpu.contacts() == referenceContacts(changed, strict), name + ": uploaded positions/radii");
    std::cout << "PASS " << name << " (" << pairs.size() << " contacts)\n";
}

void gpuTests() {
    checkScene("one sphere", {sphere(Vec3(0.f), 0.25f)});
    checkScene("separated", {sphere(Vec3(0.f), 0.25f), sphere(Vec3(2.f), 0.25f)});
    checkScene("touching", {sphere(Vec3(0.f), 0.25f), sphere(Vec3(0.5f, 0.f, 0.f), 0.25f)});
    checkScene("box corner false positive", {sphere(Vec3(0.f), 0.25f),
               sphere(Vec3(0.375f, 0.375f, 0.f), 0.25f)});
    std::vector<Particle> unequal{sphere(Vec3(0.f), 1.f),
                                  sphere(Vec3(1.05f, 0.f, 0.f), 0.1f)};
    checkScene("unilateral, large first", unequal);
    std::reverse(unequal.begin(), unequal.end());
    checkScene("unilateral, small first", unequal);
    auto boundary = std::vector<Particle>{sphere(Vec3(0.f), 0.25f),
                                          sphere(Vec3(0.25f, 0.f, 0.f), 0.125f)};
    checkScene("proxy boundary, closed", boundary);
    checkScene("proxy boundary, upstream strict", boundary, true);
    checkScene("coincident centers", {sphere(Vec3(0.f), 0.25f), sphere(Vec3(0.f), 0.125f)});
    checkScene("unequal masses and damping", {
        sphere(Vec3(0.1f), 0.25f, 0.5f, Vec3(0.2f, -0.1f, 0.4f)),
        sphere(Vec3(0.3f), 0.25f, 3.f, Vec3(-0.3f, 0.7f, -0.5f))});
    for (unsigned seed : {1u, 7u, 42u}) {
        auto random = makeScene(128, "mixed", seed);
        checkScene("random mixed seed " + std::to_string(seed), random);
        checkScene("random strict seed " + std::to_string(seed), random, true);
    }
    checkScene("dense contact buffer growth", makeScene(128, "clustered", 7));

    auto initial = makeScene(27, "mixed", 3);
    for (auto mode : {Response::Contacts, Response::Fused}) {
        for (int cadence : {0, 1, 7}) {
            Settings settings;
            Simulation gpu(initial, settings, mode);
            ReferenceSimulation cpu(initial, settings);
            for (int step = 0; step < 100; ++step) {
                gpu.step(cadence);
                cpu.step();
            }
            auto result = gpu.particles();
            for (size_t i = 0; i < result.size(); ++i) {
                expectNear(result[i].position, cpu.particles[i].position, "trajectory position", 2.e-5f);
                expectNear(result[i].velocity, cpu.particles[i].velocity, "trajectory velocity", 0.0005f);
            }
            require(gpu.contacts() == referenceContacts(result), "dynamic contacts");
        }
    }
    std::cout << "PASS trajectories: both response modes; refit, rebuild, and periodic rebuild\n";
}
}

int main(int argc, char** argv) {
    try {
        cpuTests();
        if (!(argc > 1 && std::string(argv[1]) == "--cpu-only")) gpuTests();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
