#pragma once

#include "Types.h"

namespace oops {

inline ContactParameters contactParameters(const Settings& settings) {
    float ratio = 0.f;
    if (settings.restitution <= 0.f) ratio = 2.f;
    else if (settings.restitution < 1.f) {
        float q = float(3.14159265358979323846 / std::log(settings.restitution));
        ratio = 2.f / std::sqrt(1.f + q * q);
    }
    return {settings.stiffness, ratio};
}

// Original authors' component-wise damping
inline __both__ Vec3 contactForce(const Particle& a, const Particle& b,
                                 const ContactParameters& parameters) {
    Vec3 delta = b.position - a.position;
    float distance = sqrtf(dot(delta, delta));
    float overlap = a.radius + b.radius - distance;
    if (overlap <= 0.f) return Vec3(0.f);
    Vec3 normal = distance > 0.f ? delta / distance : Vec3(1.f, 0.f, 0.f);
    float reducedMass = a.mass * b.mass / (a.mass + b.mass);
    float damping = parameters.dampingRatio * sqrtf(parameters.stiffness * reducedMass);
    Vec3 springForce = normal * overlap * parameters.stiffness;
    Vec3 dampingForce = damping * ((b.velocity - a.velocity) * normal) * normal;
    return dampingForce - springForce;
}

inline __both__ void integrateParticle(Particle& p, Vec3 force, Vec3& oldAcceleration,
                                      const Settings& settings) {
    Vec3 acceleration = force / p.mass;
    p.velocity += (acceleration + oldAcceleration) * settings.dt / 2.f;
    p.position = p.position + p.velocity * settings.dt
        + 0.5f * acceleration * settings.dt * settings.dt;
    oldAcceleration = acceleration;

    if (settings.boxWidth <= 0.f) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (p.position[axis] - p.radius < 0.f) {
            p.position[axis] = p.radius;
            p.velocity[axis] *= -settings.wallRestitution;
        } else if (p.position[axis] + p.radius - settings.boxWidth > 0.f) {
            p.position[axis] = settings.boxWidth - p.radius;
            p.velocity[axis] *= -settings.wallRestitution;
        }
    }
}

void resetForces(Particle* particles, Vec3* forces, int count, Vec3 gravity);
void applyContacts(Particle* particles, Vec3* forces, const Contact* contacts,
                   int count, const Settings& settings);
void integrate(Particle* particles, Vec3* forces, Vec3* oldAccelerations,
               int count, const Settings& settings);

} // namespace oops
