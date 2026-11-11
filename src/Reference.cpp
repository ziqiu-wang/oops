#include "Reference.h"
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
                if (p.position[axis] - p.radius < 0.f) {
                    p.position[axis] = p.radius;
                    velocity[axis] *= -settings.wallRestitution;
                } else if (p.position[axis] + p.radius - settings.boxWidth > 0.f) {
                    p.position[axis] = settings.boxWidth - p.radius;
                    velocity[axis] *= -settings.wallRestitution;
                }
            }
        }
        p.velocity = velocity;
    }
    computeForces();
}

} // namespace oops
