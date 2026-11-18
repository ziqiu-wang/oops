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

void rollingCpuTests() {
    auto s = material();
    s.physics = Physics::Normal;
    s.friction = 0.f; // Rolling must work independently of tangential friction.
    s.rollingResistance = true;
    s.rollingFriction = .1f;
    s.rollingStiffness = .2f;
    s.rollingDamping = 0.f;
    auto parameters = contactParameters(s);
    auto a = grain(Vec3(0.f, .9f, 0.f)), b = grain(Vec3(0.f));
    auto evaluate = [&](Vec3 omegaA, Vec3 omegaB, ContactState old = {}, float dt = .01f) {
        return sphereContact(a, b, omegaA, omegaB, old, s, parameters, dt);
    };
    auto first = evaluate(Vec3(0.f, 0.f, 1.f), Vec3(0.f));
    // R_eff=.25, v_roll=-.25 X, k_r=200: pseudo-force=.5 X, torque=-.125 Z.
    expectNear(first.state.rollingDisplacement, Vec3(-.0025f, 0.f, 0.f), "rolling displacement");
    expectNear(first.torqueA, Vec3(0.f, 0.f, -.125f), "rolling spring torque");
    expectNear(first.torqueA + first.torqueB, Vec3(0.f), "opposing rolling couples");
    expectNear(first.force, Vec3(0.f, 100.f, 0.f), "rolling adds no direct linear force");
    auto held = evaluate(Vec3(0.f), Vec3(0.f), first.state);
    expectNear(held.torqueA, first.torqueA, "rolling memory at rest");
    auto yielded = evaluate(Vec3(0.f, 0.f, 1000.f), Vec3(0.f));
    require(yielded.state.rollingYielding == 1, "rolling limit classification");
    expectNear(yielded.torqueA, Vec3(0.f, 0.f, -2.5f), "rolling torque cap");
    expectNear(yielded.state.rollingDisplacement, Vec3(-.05f, 0.f, 0.f), "rolling spring correction");
    s.rollingDamping = .5f;
    auto damped = evaluate(Vec3(0.f, 0.f, 1.f), Vec3(0.f), {}, 0.f);
    // Relative angular-mode mass=.8; gamma=2*.5*sqrt(200*.8).
    expectNear(damped.torqueA, Vec3(0.f, 0.f, -.7905694f), "rolling damping independent of restitution");
    require(dot(damped.torqueA, Vec3(0.f,0.f,1.f)) < 0.f, "rolling damper removes rotational energy");
    expectNear(evaluate(Vec3(0.f, 1.f, 0.f), Vec3(0.f)).torqueA, Vec3(0.f), "pure twisting is unaffected");
    expectNear(evaluate(Vec3(0.f, 0.f, 1.f), Vec3(0.f, 0.f, 1.f)).torqueA,
               Vec3(0.f), "common angular velocity is not relative rolling");
    s.rollingDamping = 0.f;
    ContactResult transported;
    transported.state.normalForce = 100.f;
    ContactState previous;
    previous.rollingDisplacement = Vec3(.01f, .01f, 0.f);
    applyRolling(transported, Vec3(0.f,1.f,0.f), .25f, 20.f, Vec3(0.f), previous, s, parameters, 0.f);
    expectNear(transported.state.rollingDisplacement, Vec3(std::sqrt(.0002f),0.f,0.f), "rolling frame transport");
    a.position.y = 2.f;
    auto separated = evaluate(Vec3(0.f,0.f,1.f), Vec3(0.f), first.state);
    expectNear(separated.torqueA, Vec3(0.f), "no rolling torque without contact");
    expectNear(separated.state.rollingDisplacement, Vec3(0.f), "separation clears rolling memory");
    a.position.y = .9f;
    s.rollingResistance = false;
    expectNear(evaluate(Vec3(0.f,0.f,1.f), Vec3(0.f), first.state).torqueA, Vec3(0.f), "rolling disabled");
    s.rollingResistance = true;
    s.rollingFriction = 0.f;
    expectNear(evaluate(Vec3(0.f,0.f,1.f), Vec3(0.f), first.state).torqueA, Vec3(0.f), "zero rolling coefficient");
    s.rollingFriction = .5f;
    s.boxWidth = 1.f;
    auto floor = wallContact(grain(Vec3(.5f,.49f,.5f)), Vec3(0.f,0.f,1.f), 2, {}, s, parameters, s.dt);
    expectNear(floor.torqueA, Vec3(0.f,0.f,-.5f), "floor rolling torque uses particle radius");
    std::cout << "PASS rolling CPU: memory, cap, damping power, torque balance, frame, walls, twisting, off switches\n";
}

void rollingStorageTests() {
    auto s = material();
    s.physics = Physics::Normal;
    s.rollingResistance = true;
    s.rollingFriction = 1.f;
    s.rollingDamping = 0.f;
    DeviceBuffer<Particle> particles;
    particles.upload({grain(Vec3(0.f,.9f,0.f)), grain(Vec3(0.f)),
                      grain(Vec3(3.f,.9f,0.f)), grain(Vec3(3.f,0.f,0.f))});
    DeviceBuffer<Vec3> omega, forces(4), torques(4);
    omega.upload({Vec3(0.f,0.f,1.f), Vec3(0.f), Vec3(0.f,0.f,2.f), Vec3(0.f)});
    DeviceBuffer<unsigned> flags(4);
    DeviceBuffer<Contact> pairs;
    ContactHistory history;
    auto apply = [&](std::vector<Contact> contacts, bool advance = true) {
        pairs.upload(contacts);
        resetGranular(particles.data(), forces.data(), torques.data(), flags.data(), 4, Vec3(0.f));
        history.apply(particles.data(), omega.data(), forces.data(), torques.data(), flags.data(),
                       pairs.data(), unsigned(contacts.size()), s, advance);
        return history.download();
    };
    apply({{0,1}});
    auto grown = apply({{2,3},{0,1}});
    require(grown[0].key == contactKey(0,1), "rolling keys are canonical and sorted");
    expectNear(grown[0].state.rollingDisplacement, Vec3(-.005f,0.f,0.f), "rolling survives growth");
    expectNear(grown[1].state.rollingDisplacement, Vec3(-.005f,0.f,0.f), "new rolling contact starts fresh");
    auto inspected = apply({{0,1}}, false);
    expectNear(inspected[0].state.rollingDisplacement, grown[0].state.rollingDisplacement, "inspection adds no rolling time");
    auto continued = apply({{2,3},{0,1}});
    expectNear(continued[1].state.rollingDisplacement, Vec3(-.01f,0.f,0.f), "inspection does not delete unqueried contacts");
    apply({{2,3}});
    auto recreated = apply({{0,1},{2,3}});
    expectNear(recreated[0].state.rollingDisplacement, Vec3(-.0025f,0.f,0.f), "rolling re-contact starts fresh");
    apply({});
    expectNear(apply({{0,1}})[0].state.rollingDisplacement, Vec3(-.0025f,0.f,0.f), "empty frame deletes rolling history");
    history.reset();
    expectNear(apply({{0,1}})[0].state.rollingDisplacement, Vec3(-.0025f,0.f,0.f), "rolling storage reset");
    std::cout << "PASS rolling GPU storage: growth, sorting, read-only subset, separation/re-contact, empty, reset\n";
}

void trajectoryTests() {
    std::vector<Particle> initial{
        {{.4f,.095f,.5f},.1f,{.15f,0.f,0.f},1.f},
        {{.4f,.285f,.5f},.1f,{-.1f,0.f,0.f},1.5f},
        {{.59f,.095f,.5f},.1f,{0.f,0.f,.05f},.8f}};
    for (auto physics : {Physics::Normal, Physics::NoHistory, Physics::History}) {
        for (bool rolling : {false, true}) {
            for (int rebuild : {0, 1, 7}) {
                Settings s;
                s.physics = physics;
                s.dt = .0001f;
                s.rollingResistance = rolling;
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
                    expectNear(c.state.rollingDisplacement, cpu.history.at(c.key).rollingDisplacement, "GPU/CPU rolling history", 2.e-6f);
                    expectNear(c.state.rollingTorque, cpu.history.at(c.key).rollingTorque, "GPU/CPU rolling torque", .002f);
                }
                auto walls = gpu.wallStates();
                for (size_t i = 0; i < walls.size(); ++i) {
                    expectNear(walls[i].rollingDisplacement, cpu.walls[i].rollingDisplacement, "GPU/CPU wall rolling history", 2.e-6f);
                    expectNear(walls[i].rollingTorque, cpu.walls[i].rollingTorque, "GPU/CPU wall rolling torque", .002f);
                }
                gpu.view = View::Preview;
                auto image = gpu.render(32, 32, 4);
                require(image == gpu.render(32, 32, 4), "frozen preview is deterministic");
                gpu.upload(initial);
                for (auto spin : gpu.angularVelocities()) expectNear(spin, Vec3(0.f), "reset angular velocity");
                for (auto c : gpu.contactStates()) expectNear(c.state.displacement, Vec3(0.f), "reset shear history");
                for (auto c : gpu.contactStates()) expectNear(c.state.rollingDisplacement, Vec3(0.f), "reset rolling history");
                for (auto c : gpu.wallStates()) expectNear(c.rollingDisplacement, Vec3(0.f), "reset wall rolling history");
            }
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

void rollingMotionTests() {
    struct Motion { float x, speed; double energy; } measured[3];
    for (int trial = 0; trial < 3; ++trial) {
        Settings s;
        s.physics = Physics::History;
        s.walls = 4;
        s.rollingResistance = trial != 0;
        s.rollingFriction = .1f;
        s.dt = trial == 2 ? .00025f : .0005f;
        std::vector<Particle> initial{{Vec3(.25f,.09877375f,.5f),.1f,Vec3(1.f,0.f,0.f),1.f}};
        Simulation gpu(initial, s);
        ReferenceGranularSimulation cpu(initial, s);
        for (int step = 0; step < int(std::lround(1.5 / s.dt)); ++step) {
            gpu.step();
            cpu.step();
        }
        auto p = gpu.particles()[0];
        expectNear(p.position, cpu.particles[0].position, "rolling motion GPU/CPU position", .0005f);
        expectNear(p.velocity, cpu.particles[0].velocity, "rolling motion GPU/CPU velocity", .005f);
        expectNear(gpu.angularVelocities()[0], cpu.omega[0], "rolling motion GPU/CPU spin", .02f);
        auto stats = gpu.diagnostics();
        measured[trial] = {p.position.x, length(p.velocity), stats.kineticEnergy};
        std::cout << "Rolling trial " << trial << ": x=" << p.position.x << ", speed="
                  << measured[trial].speed << ", energy=" << stats.kineticEnergy << '\n';
    }
    require(measured[0].speed > .5f, "sphere continues rolling when resistance is off");
    require(measured[1].speed < .03f, "rolling resistance reduces residual speed");
    require(measured[1].energy < .01 * measured[0].energy, "rolling resistance dissipates motion energy");
    require(measured[1].x - .25f < .7f * (measured[0].x - .25f), "rolling resistance reduces travel");
    require(std::abs(measured[1].x - measured[2].x) < .003f, "rolling travel converges when timestep is halved");
    require(measured[2].speed < .03f, "settling persists at the smaller timestep");
    std::cout << "PASS rolling floor motion: deceleration, shorter travel, CPU/GPU agreement, timestep check\n";
}
}

int main(int argc, char** argv) {
    try {
        cpuTests();
        rollingCpuTests();
        if (!(argc > 1 && std::string(argv[1]) == "--cpu-only")) {
            storageTests();
            rollingStorageTests();
            trajectoryTests();
            rollingMotionTests();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
