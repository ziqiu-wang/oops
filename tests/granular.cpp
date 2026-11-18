#include "ContactModel.h"
#include "Reference.h"
#include "Scene.h"
#include "Simulation.h"
#include <cmath>
#include <iostream>

using namespace oops;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void expectNear(Vec3 actual, Vec3 expected, const char* message, float tolerance = 1.e-4f) {
    for (int axis = 0; axis < 3; ++axis)
        if (!(std::abs(actual[axis] - expected[axis]) <= tolerance))
            throw std::runtime_error(std::string(message) + ": " + std::to_string(actual[axis])
                                     + " vs " + std::to_string(expected[axis]));
}
Settings material() {
    Settings s;
    s.physics = Physics::History;
    s.stiffness = 1000.f;
    s.restitution = 1.f;
    s.dt = 0.01f;
    s.gravity = Vec3(0.f);
    s.boxWidth = 0.f;
    return s;
}
Particle grain(Vec3 position, Vec3 velocity = Vec3(0.f)) {
    return {position, 0.5f, velocity, 1.f};
}

void cpuTests() {
    auto s = material();
    auto a = grain(Vec3(0.f, 0.9f, 0.f), Vec3(1.f, 0.f, 0.f));
    auto b = grain(Vec3(0.f));
    auto parameters = contactParameters(s);
    auto first = sphereContact(a, b, Vec3(0.f), Vec3(0.f), {}, s, parameters, s.dt);
    expectNear(first.force, Vec3(-4.f, 100.f, 0.f), "analytical sticking force");
    expectNear(first.torqueA, Vec3(0.f, 0.f, -1.8f), "contact torque A");
    expectNear(first.torqueB, first.torqueA, "contact torque B");
    a.velocity = Vec3(0.f);
    auto held = sphereContact(a, b, Vec3(0.f), Vec3(0.f), first.state, s, parameters, s.dt);
    expectNear(held.state.tangentialForce, Vec3(-4.f, 0.f, 0.f), "static contact memory");
    s.physics = Physics::NoHistory;
    auto noHistory = sphereContact(a, b, Vec3(0.f), Vec3(0.f), first.state, s, parameters, s.dt);
    expectNear(noHistory.state.tangentialForce, Vec3(0.f), "no-history force at rest");
    s.physics = Physics::History;
    a.velocity = Vec3(100.f, 0.f, 0.f);
    auto sliding = sphereContact(a, b, Vec3(0.f), Vec3(0.f), {}, s, parameters, s.dt);
    require(sliding.state.sliding == 1, "sliding classification");
    expectNear(sliding.state.tangentialForce, Vec3(-50.f, 0.f, 0.f), "Coulomb cap");
    expectNear(sliding.state.displacement, Vec3(0.125f, 0.f, 0.f), "spring correction after sliding");
    a.position.y = 2.f;
    auto separated = sphereContact(a, b, Vec3(0.f), Vec3(0.f), first.state, s, parameters, s.dt);
    expectNear(separated.state.displacement, Vec3(0.f), "separation clears state");
    ContactState old;
    old.displacement = Vec3(0.01f, 0.01f, 0.f);
    auto rotated = evaluateContact(Vec3(0.f, 1.f, 0.f), 0.1f, 0.5f, Vec3(0.f),
        Vec3(0.f), 0.5f, 0.5f, old, s, parameters, 0.f);
    expectNear(rotated.state.displacement, Vec3(std::sqrt(0.0002f), 0.f, 0.f), "history frame transport");
    s.boxWidth = 1.f;
    a = grain(Vec3(0.5f, 0.49f, 0.5f), Vec3(1.f, 0.f, 0.f));
    auto rolling = wallContact(a, Vec3(0.f, 0.f, -2.f), 2, {}, s, parameters, s.dt);
    expectNear(rolling.state.tangentialForce, Vec3(0.f), "pure rolling has no surface slip");
    s.physics = Physics::Normal;
    a.velocity = Vec3(0.f);
    auto wall = wallContact(a, Vec3(0.f), 2, {}, s, parameters, s.dt);
    expectNear(wall.force, Vec3(0.f, 10.f, 0.f), "plane spring force");
    std::cout << "PASS granular CPU: sticking, held force, sliding, transport, torque, walls, rolling\n";
}

void storageTests() {
    auto s = material();
    std::vector<Particle> initial{grain(Vec3(0.f, .9f, 0.f), Vec3(1.f, 0.f, 0.f)), grain(Vec3(0.f)),
        grain(Vec3(3.f, .9f, 0.f), Vec3(2.f, 0.f, 0.f)), grain(Vec3(3.f, 0.f, 0.f))};
    DeviceBuffer<Particle> particles;
    particles.upload(initial);
    DeviceBuffer<Vec3> omega(4), forces(4), torques(4);
    DeviceBuffer<unsigned> flags(4);
    DeviceBuffer<Contact> pairs;
    omega.clear();
    ContactHistory history;
    auto apply = [&](std::vector<Contact> contacts, bool advance = true) {
        pairs.upload(contacts);
        resetGranular(particles.data(), forces.data(), torques.data(), flags.data(), 4, Vec3(0.f));
        history.apply(particles.data(), omega.data(), forces.data(), torques.data(), flags.data(),
                       pairs.data(), unsigned(contacts.size()), s, advance);
        return history.download();
    };
    apply({{0, 1}});
    auto grown = apply({{2, 3}, {0, 1}});
    require(grown[0].key == contactKey(0, 1), "GPU pair sort");
    expectNear(grown[0].state.displacement, Vec3(.02f, 0.f, 0.f), "history survives growth");
    expectNear(grown[1].state.displacement, Vec3(.02f, 0.f, 0.f), "new contact starts at zero");
    auto inspected = apply({{0, 1}, {2, 3}}, false);
    expectNear(inspected[0].state.displacement, grown[0].state.displacement, "read-only evaluation");
    apply({{2, 3}});
    auto recreated = apply({{0, 1}, {2, 3}});
    expectNear(recreated[0].state.displacement, Vec3(.01f, 0.f, 0.f), "separation and re-contact");
    apply({});
    auto fresh = apply({{0, 1}});
    expectNear(fresh[0].state.displacement, Vec3(.01f, 0.f, 0.f), "empty contact frame clears history");
    history.reset();
    auto reset = apply({{0, 1}});
    expectNear(reset[0].state.displacement, Vec3(.01f, 0.f, 0.f), "explicit reset");
    std::cout << "PASS GPU storage: sorting, growth, read-only evaluation, deletion, re-contact, reset\n";
}

void trajectoryTests() {
    std::vector<Particle> initial{
        {{.4f,.095f,.5f},.1f,{.15f,0.f,0.f},1.f},
        {{.4f,.285f,.5f},.1f,{-.1f,0.f,0.f},1.5f},
        {{.59f,.095f,.5f},.1f,{0.f,0.f,.05f},.8f}};
    for (auto physics : {Physics::Normal, Physics::NoHistory, Physics::History}) {
        for (int rebuild : {0, 1, 7}) {
            Settings s;
            s.physics = physics;
            s.dt = .0001f;
            Simulation gpu(initial, s, Response::Contacts, 1);
            ReferenceGranularSimulation cpu(initial, s);
            for (int step = 0; step < 40; ++step) {
                gpu.step(rebuild);
                cpu.step();
                if (step % 3 == 0) {
                    gpu.computeForces();
                    gpu.contacts();
                }
            }
            auto actual = gpu.particles();
            auto angular = gpu.angularVelocities();
            auto forces = gpu.forces();
            for (size_t i = 0; i < initial.size(); ++i) {
                expectNear(actual[i].position, cpu.particles[i].position, "GPU/CPU position", 3.e-6f);
                expectNear(actual[i].velocity, cpu.particles[i].velocity, "GPU/CPU velocity", 3.e-4f);
                expectNear(angular[i], cpu.omega[i], "GPU/CPU spin", .003f);
                expectNear(forces[i], cpu.forces[i], "GPU/CPU force", .02f);
            }
            for (auto c : gpu.contactStates()) {
                expectNear(c.state.displacement, cpu.history.at(c.key).displacement, "GPU/CPU history", 2.e-6f);
                expectNear(c.state.tangentialForce, cpu.history.at(c.key).tangentialForce, "GPU/CPU friction", .02f);
            }
            gpu.view = View::Preview;
            auto image = gpu.render(32, 32, 4);
            require(image == gpu.render(32, 32, 4), "frozen preview is deterministic");
            gpu.upload(initial);
            for (auto spin : gpu.angularVelocities()) expectNear(spin, Vec3(0.f), "reset angular velocity");
            for (auto c : gpu.contactStates()) expectNear(c.state.displacement, Vec3(0.f), "reset shear history");
        }
    }
    // A free particle is an independent check of kick-drift-kick initialization.
    auto s = material();
    s.physics = Physics::Normal;
    s.gravity = Vec3(0.f, -10.f, 0.f);
    Simulation falling({grain(Vec3(0.f))}, s);
    falling.step();
    auto p = falling.particles()[0];
    expectNear(p.position, Vec3(0.f, -.0005f, 0.f), "analytical gravity position");
    expectNear(p.velocity, Vec3(0.f, -.1f, 0.f), "analytical gravity velocity");
    std::cout << "PASS granular GPU/CPU trajectories: all models and BVH cadences; inspection and rendering; gravity\n";
}
}

int main(int argc, char** argv) {
    try {
        cpuTests();
        if (!(argc > 1 && std::string(argv[1]) == "--cpu-only")) {
            storageTests();
            trajectoryTests();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
