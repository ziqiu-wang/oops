#include "Simulation.h"
#include "Granular.h"
#include <cuda_runtime.h>
#include <chrono>

namespace oops
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        double elapsed(Clock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        void syncIf(bool on)
        {
            if (on)
                cudaDeviceSynchronize();
        }
    }

    Simulation::Simulation(const std::vector<Particle> &particles, Settings configuration,
                           Response mode, unsigned capacity)
        : settings(configuration), response(mode), forces_(particles.size()),
          oldAccelerations_(particles.size())
    {
        if (particles.empty())
            throw std::runtime_error("A simulation needs at least one particle.");
        particles_.upload(particles);
        forces_.clear();
        oldAccelerations_.clear();
        backend_ = std::make_unique<OptixBackend>(particles_.data(), int(particles.size()), capacity);
    }

    void Simulation::computeForces()
    {
        resetForces(particles_.data(), forces_.data(), int(particles_.size()), settings.gravity);
        syncIf(profile);
        auto start = Clock::now();
        if (response == Response::Fused)
        {
            backend_->accumulateForces(forces_.data(), settings);
            syncIf(profile);
            timings.query += elapsed(start);
        }
        else
        {
            unsigned count = backend_->findContacts(settings);
            syncIf(profile);
            timings.query += elapsed(start);
            start = Clock::now();
            applyContacts(particles_.data(), forces_.data(), backend_->deviceContacts(), count, settings);
            syncIf(profile);
            timings.response += elapsed(start);
        }
    }

    void Simulation::step(int rebuildEvery)
    {
        auto start = Clock::now();
        integrate(particles_.data(), forces_.data(), oldAccelerations_.data(),
                  int(particles_.size()), settings);
        syncIf(profile);
        timings.integrate += elapsed(start);
        ++steps;
        start = Clock::now();
        backend_->update(rebuildEvery > 0 && steps % rebuildEvery == 0);
        syncIf(profile);
        timings.accel += elapsed(start);
        computeForces();
    }

    void Simulation::upload(const std::vector<Particle> &particles)
    {
        if (particles.size() != particles_.size())
            throw std::runtime_error("upload() keeps particle IDs and count fixed.");
        particles_.upload(particles);
        forces_.clear();
        oldAccelerations_.clear();
        backend_->update(true);
        steps = 0;
        timings = {};
    }

    std::vector<Contact> Simulation::contacts()
    {
        backend_->findContacts(settings);
        return backend_->contacts();
    }

    OrbitCamera Simulation::defaultCamera() const
    {
        float scale = settings.boxWidth > 0.f ? settings.boxWidth : 1.f;
        Vec3 target = scale * Vec3(0.5f, 0.35f, 0.5f);
        Vec3 eye = scale * Vec3(1.8f, 1.35f, 1.95f);
        return OrbitCamera(eye, target, 0.45f);
    }

    std::vector<uint32_t> Simulation::render(int width, int height, int samples)
    {
        return backend_->render(defaultCamera().camera(), width, height, samples);
    }

    void Simulation::renderTo(uint32_t *devicePixels, const Camera &camera,
                              int width, int height, int samples)
    {
        backend_->renderTo(camera, devicePixels, width, height, samples);
    }

} // namespace oops
