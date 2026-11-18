#pragma once

#include "Types.h"
#include <vector>
#include <map>

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

// All-pairs CPU driver for short granular trajectories; no GPU storage or sorting.
struct ReferenceGranularSimulation {
    std::vector<Particle> particles;
    std::vector<Vec3> forces, omega, torques;
    std::vector<ContactState> walls;
    std::map<uint64_t, ContactState> history;
    Settings settings;
    ReferenceGranularSimulation(std::vector<Particle> input, Settings configuration);
    void computeForces();
    void step();
private:
    bool ready_ = false;
    void evaluate(bool advance);
    void kick(bool drift);
};

} // namespace oops
