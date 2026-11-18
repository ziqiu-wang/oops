#pragma once

#include "CudaBuffer.h"
#include "Types.h"

namespace oops {

class ContactHistory {
public:
    void apply(const Particle* particles, const Vec3* omega, Vec3* forces, Vec3* torques,
               unsigned* flags, const Contact* contacts, unsigned count, const Settings& settings, bool advance);
    void reset() { count_ = viewCount_ = 0; }
    std::vector<StoredContact> download() const;
    size_t bytes() const;

private:
    DeviceBuffer<uint64_t> previousKeys_, nextKeys_;
    DeviceBuffer<ContactState> previous_, next_;
    unsigned count_ = 0, viewCount_ = 0;
    bool viewPrevious_ = false;
};

void resetGranular(const Particle* particles, Vec3* forces, Vec3* torques, unsigned* flags,
                   int count, Vec3 gravity);
void kickDrift(Particle* particles, Vec3* omega, const Vec3* forces, const Vec3* torques,
               int count, float dt, bool drift);
void applyWalls(const Particle* particles, const Vec3* omega, Vec3* forces, Vec3* torques,
                unsigned* flags, ContactState* history, int count, const Settings& settings, bool advance);

} // namespace oops
