#pragma once

#include <owl/common/math/vec.h>
#include <cstdint>
#include <cstddef>

namespace oops {

using Vec3 = owl::common::vec3f;

// Array indices are particle IDs so do not reorder during a simulation
struct Particle {
    Vec3 position{0.f};
    float radius{0.01f};
    Vec3 velocity{0.f};
    float mass{1.f};
};

struct Contact {
    uint32_t a, b;
};

enum class Physics { Mochi, Normal, NoHistory, History };
enum class View { Path, Preview, Speed, Contacts };

struct ContactState {
    Vec3 displacement{0.f};
    Vec3 tangentialForce{0.f};
    float normalForce = 0.f;
    unsigned sliding = 0;
};

struct StoredContact {
    uint64_t key;
    ContactState state;
};

struct Diagnostics {
    unsigned contacts = 0, sliding = 0;
    float maximumSpeed = 0.f;
    double kineticEnergy = 0.;
    size_t historyBytes = 0;
};

struct Settings {
    float dt = 0.0001f;
    float stiffness = 8000.f;
    float restitution = 0.5f;
    float wallRestitution = 0.1f;
    float boxWidth = 1.f;  // <= 0 disables walls
    Vec3 gravity{0.f, -9.81f, 0.f};
    bool strictUpstream = false;  // original strict proxy/overlap inequalities
    Physics physics = Physics::Mochi;
    float friction = 0.5f;
    float tangentialStiffness = 0.4f; // k_t / k_n
    float tangentialDamping = 1.f;   // eta_t / eta_n
    unsigned walls = 63; // bits: -X,+X,-Y,+Y,-Z,+Z; floor only = 4
};

enum class Response { Fused, Contacts };

struct ContactParameters {
    float stiffness;
    float dampingRatio;
};

struct Timings {
    double integrate = 0;
    double accel = 0;
    double query = 0;
    double response = 0;
};

inline bool operator<(Contact x, Contact y) {
    return x.a < y.a || (x.a == y.a && x.b < y.b);
}

inline bool operator==(Contact x, Contact y) {
    return x.a == y.a && x.b == y.b;
}

} // namespace oops
