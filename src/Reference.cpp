#include "Reference.h"
#include "ContactModel.h"
#include <cmath>

namespace oops {

std::vector<Contact> referenceContacts(const std::vector<Particle>& particles, bool strict) {
    std::vector<Contact> pairs;
    for (unsigned i = 0; i < particles.size(); ++i) {
        for (unsigned j = i + 1; j < particles.size(); ++j) {
            const auto& a = particles[i];
            const auto& b = particles[j];
            Vec3 diff = b.position - a.position;
            float squared = dot(diff, diff);
            if (!strict) {
                float radius = a.radius + b.radius;
                if (squared <= radius * radius) pairs.push_back({i, j});
            } else {
                float distance = std::sqrt(squared);

                bool forward = 2.f * b.radius - distance > 0.f;
                bool reverse = 2.f * a.radius - distance > 0.f
                    && 2.f * b.radius - distance < 0.f;
                if (a.radius + b.radius - distance > 0.f && (forward || reverse))
                    pairs.push_back({i, j});
            }
        }
    }
    return pairs;
}

std::vector<Vec3> referenceForces(const std::vector<Particle>& particles, const Settings& settings) {
    if (settings.physics != Physics::Mochi) {
        ReferenceGranularSimulation simulation(particles, settings);
        simulation.computeForces();
        return simulation.forces;
    }
    std::vector<Vec3> forces;
    for (auto p : particles) forces.push_back(p.mass * settings.gravity);
    float q = float(3.14159265358979323846 / std::log(settings.restitution));
    float ratio = settings.restitution >= 1.f ? 0.f
        : settings.restitution <= 0.f ? 2.f
        : 2.f / std::sqrt(1.f + q * q);
    for (auto pair : referenceContacts(particles, settings.strictUpstream)) {
        const auto& a = particles[pair.a];
        const auto& b = particles[pair.b];
        Vec3 difference = b.position - a.position;
        float distance = std::sqrt(dot(difference, difference));
        float overlap = a.radius + b.radius - distance;
        if (overlap <= 0.f) continue;
        Vec3 normal = distance > 0.f ? difference / distance : Vec3(1.f, 0.f, 0.f);
        float coefficient = ratio * std::sqrt(settings.stiffness * a.mass * b.mass / (a.mass + b.mass));
        for (int axis = 0; axis < 3; ++axis) {
            float force = coefficient * (b.velocity[axis] - a.velocity[axis])
                * normal[axis] * normal[axis] - settings.stiffness * overlap * normal[axis];
            forces[pair.a][axis] += force;
            forces[pair.b][axis] -= force;
        }
    }
    return forces;
}

ReferenceSimulation::ReferenceSimulation(std::vector<Particle> input, Settings configuration)
    : particles(std::move(input)), forces(particles.size(), Vec3(0.f)),
      previous(particles.size(), Vec3(0.f)), settings(configuration) {}

void ReferenceSimulation::computeForces() { forces = referenceForces(particles, settings); }

void ReferenceSimulation::step() {
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        Vec3 acceleration = forces[i] / p.mass;
        Vec3 velocity = p.velocity + (acceleration + previous[i]) * settings.dt / 2.f;
        p.position = p.position + velocity * settings.dt
            + 0.5f * acceleration * settings.dt * settings.dt;
        previous[i] = acceleration;
        if (settings.boxWidth > 0.f) {
            for (int axis = 0; axis < 3; ++axis) {
                if ((settings.walls & (1u << (2 * axis))) && p.position[axis] - p.radius < 0.f) {
                    p.position[axis] = p.radius;
                    velocity[axis] *= -settings.wallRestitution;
                } else if ((settings.walls & (1u << (2 * axis + 1)))
                           && p.position[axis] + p.radius - settings.boxWidth > 0.f) {
                    p.position[axis] = settings.boxWidth - p.radius;
                    velocity[axis] *= -settings.wallRestitution;
                }
            }
        }
        p.velocity = velocity;
    }
    computeForces();
}

ReferenceGranularSimulation::ReferenceGranularSimulation(std::vector<Particle> input, Settings configuration)
    : particles(std::move(input)), forces(particles.size()), omega(particles.size(), Vec3(0.f)),
      torques(particles.size()), walls(6 * particles.size()), settings(configuration) {}

void ReferenceGranularSimulation::computeForces() {
    if (!ready_) evaluate(false);
}

void ReferenceGranularSimulation::evaluate(bool advance) {
    for (size_t i = 0; i < particles.size(); ++i) {
        forces[i] = particles[i].mass * settings.gravity;
        torques[i] = Vec3(0.f);
    }
    auto parameters = contactParameters(settings);
    std::map<uint64_t, ContactState> next;
    for (auto pair : referenceContacts(particles, settings.strictUpstream)) {
        uint64_t key = contactKey(pair.a, pair.b);
        auto found = history.find(key);
        ContactState old = found == history.end() ? ContactState{} : found->second;
        auto result = sphereContact(particles[pair.a], particles[pair.b], omega[pair.a], omega[pair.b],
                                    old, settings, parameters, advance ? settings.dt : 0.f);
        forces[pair.a] += result.force;
        forces[pair.b] -= result.force;
        torques[pair.a] += result.torqueA;
        torques[pair.b] += result.torqueB;
        next[key] = result.state;
    }
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int wall = 0; wall < 6; ++wall) {
            auto result = wallContact(particles[i], omega[i], wall, walls[6 * i + wall],
                                      settings, parameters, advance ? settings.dt : 0.f);
            forces[i] += result.force;
            torques[i] += result.torqueA;
            if (advance) walls[6 * i + wall] = result.state;
        }
    }
    if (advance) history.swap(next);
    ready_ = true;
}

void ReferenceGranularSimulation::kick(bool drift) {
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        p.velocity += (0.5f * settings.dt / p.mass) * forces[i];
        omega[i] += (0.5f * settings.dt * inverseInertia(p)) * torques[i];
        if (drift) p.position += settings.dt * p.velocity;
    }
}

void ReferenceGranularSimulation::step() {
    computeForces();
    kick(true);
    evaluate(true);
    kick(false);
}

} // namespace oops
