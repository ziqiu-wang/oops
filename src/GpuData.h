#pragma once

#include "Types.h"
#include <optix_types.h>

namespace oops {

struct GeometryData { Particle* particles; };

struct Camera {
    Vec3 origin, forward, right, up;
};

struct LaunchData {
    OptixTraversableHandle world = 0;
    Particle* particles = nullptr;
    Vec3* forces = nullptr;
    Contact* contacts = nullptr;
    unsigned* contactCount = nullptr;
    unsigned capacity = 0;
    bool emitContacts = false;
    Settings settings;
    ContactParameters contact;
    uint32_t* pixels = nullptr;
    Camera camera;
    int samples = 4;
};

} // namespace oops
