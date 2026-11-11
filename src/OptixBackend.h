#pragma once

#include "CudaBuffer.h"
#include "GpuData.h"
#include <memory>
#include <vector>

namespace oops {

class OptixBackend {
public:
    OptixBackend(Particle* particles, int count, unsigned contactCapacity = 0);
    ~OptixBackend();
    OptixBackend(const OptixBackend&) = delete;
    OptixBackend& operator=(const OptixBackend&) = delete;

    void update(bool rebuild);
    unsigned findContacts(const Settings& settings);
    void accumulateForces(Vec3* forces, const Settings& settings);
    std::vector<Contact> contacts() const;
    const Contact* deviceContacts() const { return contacts_.data(); }
    std::vector<uint32_t> render(const Camera& camera, int width, int height, int samples);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DeviceBuffer<Contact> contacts_;
    DeviceBuffer<unsigned> counter_{1};
    DeviceBuffer<uint32_t> pixels_;
    LaunchData launch_;
    int particleCount_;
    unsigned initialCapacity_;
    unsigned contactCount_ = 0;
};

} // namespace oops
