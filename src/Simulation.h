#pragma once

#include "OptixBackend.h"
#include "OrbitCamera.h"
#include "ContactHistory.h"
#include "GpuTimer.h"
#include "bvhOptimizer.h"
#include <memory>

namespace oops {

class Simulation {
public:
    Simulation(const std::vector<Particle>& particles, Settings settings = {},
               Response response = Response::Contacts, unsigned contactCapacity = 0);
    void computeForces();
    void step(int rebuildEvery = 0); // 0 = refit only, N = rebuild every N steps, -1 = adaptive
    void upload(const std::vector<Particle>& particles); // Same count; resets dynamics.
    std::vector<Particle> particles() const { return particles_.download(); }
    std::vector<Vec3> forces() const { return forces_.download(); }
    std::vector<Contact> contacts();
    std::vector<uint32_t> render(int width = 800, int height = 600, int samples = 4);
    void renderTo(uint32_t* devicePixels, const Camera& camera, int width = 800, int height = 600, int samples = 4);
    OrbitCamera defaultCamera() const;
    std::vector<Vec3> angularVelocities() const { return omega_.download(); }
    std::vector<Vec3> torques() const { return torques_.download(); }
    std::vector<StoredContact> contactStates() const { return history_.download(); }
    std::vector<ContactState> wallStates() const { return wallHistory_.download(); }
    Diagnostics diagnostics();
    void saveGranularState(const std::string& path) const;
    View view = View::Path;

    Settings settings;
    Response response;
    Timings timings;
    int steps = 0;

    bool profile = false;

private:
    DeviceBuffer<Particle> particles_;
    DeviceBuffer<Vec3> forces_, oldAccelerations_;
    DeviceBuffer<Vec3> omega_, torques_;
    DeviceBuffer<unsigned> contactFlags_;
    DeviceBuffer<ContactState> wallHistory_;
    ContactHistory history_;
    bool forcesReady_ = false;
    void granularForces(bool advance);
    void updateAccel(int rebuildEvery);
    void beginQueryTimer();
    void endQueryTimer();
    void reportBvhCosts();

    // Adaptive BVH schedule. The GPU timers force a sync, so they only run when it is enabled.
    GpuTimer accelGpu_, queryGpu_;
    bvhOptimizer bvhOptimizer_;
    bool adaptiveBvh_ = false, lastRebuilt_ = false;
    float lastAccelMs_ = 0.f, lastQueryMs_ = 0.f;
    std::unique_ptr<OptixBackend> backend_;
};

void writeImage(const std::string& path, int width, int height,
                const std::vector<uint32_t>& pixels);

} // namespace oops
