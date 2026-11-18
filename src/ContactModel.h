#pragma once

#include "Granular.h"

namespace oops {

inline __both__ uint64_t contactKey(unsigned a, unsigned b) {
    return (uint64_t(a) << 32) | b; // caller supplies a < b
}

struct ContactResult {
    Vec3 force{0.f}, torqueA{0.f}, torqueB{0.f};
    ContactState state;
};

// n points from B toward A. Relative surface velocity is A minus B.
inline __both__ ContactResult evaluateContact(Vec3 n, float overlap, float reducedMass,
    Vec3 relativeVelocity, Vec3 rotationalVelocity, float armA, float armB,
    ContactState previous, const Settings& settings, ContactParameters parameters, float dt) {
    ContactResult result;
    if (overlap <= 0.f) return result;
    float normalVelocity = dot(relativeVelocity, n);
    float damping = parameters.dampingRatio * sqrtf(parameters.stiffness * reducedMass);
    float normalForce = fmaxf(0.f, parameters.stiffness * overlap - damping * normalVelocity);
    Vec3 tangentVelocity = relativeVelocity - normalVelocity * n - cross(rotationalVelocity, n);
    result.state.normalForce = normalForce;
    result.force = normalForce * n;
    if (settings.physics == Physics::Normal || settings.friction == 0.f) return result;

    float stiffness = settings.tangentialStiffness * parameters.stiffness;
    float tangentDamping = settings.tangentialDamping * damping;
    Vec3 displacement(0.f);
    if (settings.physics == Physics::History) {
        // Transport the elastic displacement into the current contact plane.
        float oldLength = sqrtf(dot(previous.displacement, previous.displacement));
        displacement = previous.displacement - dot(previous.displacement, n) * n;
        float newLength = sqrtf(dot(displacement, displacement));
        if (newLength > 1.e-20f) displacement *= oldLength / newLength;
        displacement += tangentVelocity * dt;
    }
    Vec3 dampingForce = -tangentDamping * tangentVelocity;
    Vec3 tangentForce = -stiffness * displacement + dampingForce;
    float magnitude = sqrtf(dot(tangentForce, tangentForce));
    float limit = settings.friction * normalForce;
    if (magnitude > limit) {
        tangentForce *= limit / magnitude;
        result.state.sliding = 1;
        if (settings.physics == Physics::History)
            displacement = (dampingForce - tangentForce) / stiffness;
    }
    result.state.displacement = displacement;
    result.state.tangentialForce = tangentForce;
    result.force += tangentForce;
    Vec3 moment = -cross(n, tangentForce);
    result.torqueA = armA * moment;
    result.torqueB = armB * moment;
    return result;
}

inline __both__ ContactResult sphereContact(const Particle& a, const Particle& b,
    Vec3 omegaA, Vec3 omegaB, ContactState previous, const Settings& settings,
    ContactParameters parameters, float dt) {
    Vec3 delta = a.position - b.position;
    float distance = sqrtf(dot(delta, delta));
    Vec3 normal = distance > 0.f ? delta / distance : Vec3(1.f, 0.f, 0.f);
    float overlap = a.radius + b.radius - distance;
    return evaluateContact(normal, overlap, a.mass * b.mass / (a.mass + b.mass),
        a.velocity - b.velocity, a.radius * omegaA + b.radius * omegaB,
        a.radius - 0.5f * overlap, b.radius - 0.5f * overlap,
        previous, settings, parameters, dt);
}

inline __both__ ContactResult wallContact(const Particle& p, Vec3 omega, int wall,
    ContactState previous, const Settings& settings, ContactParameters parameters, float dt) {
    Vec3 normal(0.f);
    int axis = wall / 2;
    normal[axis] = wall % 2 ? -1.f : 1.f;
    float distance = wall % 2 ? settings.boxWidth - p.position[axis] : p.position[axis];
    if (!(settings.walls & (1u << wall)) || settings.boxWidth <= 0.f) return {};
    return evaluateContact(normal, p.radius - distance, p.mass, p.velocity,
        p.radius * omega, p.radius, 0.f, previous, settings, parameters, dt);
}

inline __both__ float inverseInertia(const Particle& p) {
    return 2.5f / (p.mass * p.radius * p.radius);
}

} // namespace oops
