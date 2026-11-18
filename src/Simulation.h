#pragma once

#include "OptixBackend.h"
#include "OrbitCamera.h"
#include "GpuTimer.h"
#include "bvhOptimizer.h"
#include <memory>

namespace oops {

class Simulation {
public:
    Simulation(const std::vector<Particle>& particles, Settings settings = {},
               Response response = Response::Contacts, unsigned contactCapacity = 0);
    void computeForces();
    void step(int rebuildEvery = 0);
    void upload(const std::vector<Particle>& particles); // Same count; resets dynamics.
    std::vector<Particle> particles() const { return particles_.download(); }
    std::vector<Vec3> forces() const { return forces_.download(); }
    std::vector<Contact> contacts();
    std::vector<uint32_t> render(int width = 800, int height = 600, int samples = 4);
    void renderTo(uint32_t* devicePixels, const Camera& camera, int width = 800, int height = 600, int samples = 4);
    OrbitCamera defaultCamera() const;

    Settings settings;
    Response response;
    Timings timings;
    int steps = 0;

    bool profile = false;

private:
    DeviceBuffer<Particle> particles_;
    DeviceBuffer<Vec3> forces_, oldAccelerations_;
    std::unique_ptr<OptixBackend> backend_;
    GpuTimer accelGpu, queryGpu;
    bvhOptimizer bvhOpt;
    float lastQueryTime;
};

void writeImage(const std::string& path, int width, int height,
                const std::vector<uint32_t>& pixels);

} // namespace oops
