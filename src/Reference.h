#pragma once

#include "Types.h"
#include <vector>

namespace oops {

std::vector<Contact> referenceContacts(const std::vector<Particle>& particles, bool strict = false);
std::vector<Vec3> referenceForces(const std::vector<Particle>& particles, const Settings& settings);

struct ReferenceSimulation {
    std::vector<Particle> particles;
    std::vector<Vec3> forces, previous;
    Settings settings;
    ReferenceSimulation(std::vector<Particle> input, Settings configuration = {});
    void computeForces();
    void step();
};

} // namespace oops
