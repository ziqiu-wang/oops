#pragma once

#include "CudaBuffer.h"
#include "GpuData.h"
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace oops {

class OptixBackend {
public:
    OptixBackend(Particle* particles, int count, unsigned contactCapacity = 0);
    ~OptixBackend();
    OptixBackend(const OptixBackend&) = delete;
    OptixBackend& operator=(const OptixBackend&) = delete;

    void update(bool rebuild);
    void waitForCuda(); // Order default-stream physics before OWL's two streams.
    void setView(View view, const unsigned* contactFlags = nullptr);
    unsigned findContacts(const Settings& settings);
    void accumulateForces(Vec3* forces, const Settings& settings);
    std::vector<Contact> contacts() const;
    const Contact* deviceContacts() const { return contacts_.data(); }
    std::vector<uint32_t> render(const Camera& camera, int width, int height, int samples);
    void renderTo(const Camera& camera, uint32_t* devicePixels, int width, int height, int samples);

    // stream for gpu timer measuring bvh rebuild and refit 
    cudaStream_t defaultStream() const { return nullptr; }

    // stream for query and render launches 
    cudaStream_t launchStream() const;

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
