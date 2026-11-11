#pragma once

#include <owl/common/math/vec.h>
#include <cstdint>

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

struct Settings {
    float dt = 0.0001f;
    float stiffness = 8000.f;
    float restitution = 0.5f;
    float wallRestitution = 0.1f;
    float boxWidth = 1.f;  // <= 0 disables walls
    Vec3 gravity{0.f, -9.81f, 0.f};
    bool strictUpstream = false;  // original strict proxy/overlap inequalities
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
