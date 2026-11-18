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

inline __both__ Vec3 transportDisplacement(Vec3 displacement, Vec3 normal) {
    float oldLength = sqrtf(dot(displacement, displacement));
    displacement -= dot(displacement, normal) * normal;
    float newLength = sqrtf(dot(displacement, displacement));
    if (newLength > 1.e-20f) displacement *= oldLength / newLength;
    return displacement;
}

inline __both__ float inverseInertia(const Particle& p) {
    return 2.5f / (p.mass * p.radius * p.radius);
}

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
        displacement = transportDisplacement(previous.displacement, n);
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

// LAMMPS SDS convention: v_roll = R_eff * ((omegaA - omegaB) cross n).
// The pseudo-force changes only torque, never the linear force accumulator.
inline __both__ void applyRolling(ContactResult& result, Vec3 normal, float radius,
    float inverseInertiaSum, Vec3 relativeOmega, ContactState previous,
    const Settings& settings, ContactParameters parameters, float dt) {
    if (!settings.rollingResistance || settings.rollingFriction == 0.f
        || result.state.normalForce <= 0.f) return;
    Vec3 velocity = radius * cross(relativeOmega, normal);
    Vec3 displacement = transportDisplacement(previous.rollingDisplacement, normal) + dt * velocity;
    float stiffness = settings.rollingStiffness * parameters.stiffness;
    // Effective mass for the relative angular mode expressed in rolling-length units.
    float mass = 1.f / (radius * radius * inverseInertiaSum);
    float damping = 2.f * settings.rollingDamping * sqrtf(stiffness * mass);
    Vec3 dampingForce = -damping * velocity;
    Vec3 pseudoForce = -stiffness * displacement + dampingForce;
    float magnitude = sqrtf(dot(pseudoForce, pseudoForce));
    float limit = settings.rollingFriction * result.state.normalForce;
    if (magnitude > limit) {
        pseudoForce *= limit / magnitude;
        displacement = (dampingForce - pseudoForce) / stiffness;
        result.state.rollingYielding = 1;
    }
    Vec3 torque = radius * cross(normal, pseudoForce);
    result.state.rollingDisplacement = displacement;
    result.state.rollingTorque = torque;
    result.torqueA += torque;
    result.torqueB -= torque;
}

inline __both__ ContactResult sphereContact(const Particle& a, const Particle& b,
    Vec3 omegaA, Vec3 omegaB, ContactState previous, const Settings& settings,
    ContactParameters parameters, float dt) {
    Vec3 delta = a.position - b.position;
    float distance = sqrtf(dot(delta, delta));
    Vec3 normal = distance > 0.f ? delta / distance : Vec3(1.f, 0.f, 0.f);
    float overlap = a.radius + b.radius - distance;
    auto result = evaluateContact(normal, overlap, a.mass * b.mass / (a.mass + b.mass),
        a.velocity - b.velocity, a.radius * omegaA + b.radius * omegaB,
        a.radius - 0.5f * overlap, b.radius - 0.5f * overlap,
        previous, settings, parameters, dt);
    if (settings.rollingResistance)
        applyRolling(result, normal, a.radius * b.radius / (a.radius + b.radius),
                     inverseInertia(a) + inverseInertia(b), omegaA - omegaB,
                     previous, settings, parameters, dt);
    return result;
}

inline __both__ ContactResult wallContact(const Particle& p, Vec3 omega, int wall,
    ContactState previous, const Settings& settings, ContactParameters parameters, float dt) {
    Vec3 normal(0.f);
    int axis = wall / 2;
    normal[axis] = wall % 2 ? -1.f : 1.f;
    float distance = wall % 2 ? settings.boxWidth - p.position[axis] : p.position[axis];
    if (!(settings.walls & (1u << wall)) || settings.boxWidth <= 0.f) return {};
    auto result = evaluateContact(normal, p.radius - distance, p.mass, p.velocity,
        p.radius * omega, p.radius, 0.f, previous, settings, parameters, dt);
    if (settings.rollingResistance)
        applyRolling(result, normal, p.radius, inverseInertia(p), omega,
                     previous, settings, parameters, dt);
    return result;
}

} // namespace oops
