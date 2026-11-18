#include "Simulation.h"
#include "Granular.h"
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>

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
        if (settings.physics != Physics::Mochi) upload(particles);
    }

    void Simulation::computeForces()
    {
        if (settings.physics != Physics::Mochi) {
            // Preserve cached half-step forces. Inspection must not change the next kick.
            if (!forcesReady_) granularForces(false);
            return;
        }
        resetForces(particles_.data(), forces_.data(), int(particles_.size()), settings.gravity);
        syncIf(profile);
        auto start = Clock::now();
        beginQueryTimer();
        if (response == Response::Fused)
        {
            backend_->accumulateForces(forces_.data(), settings);
            endQueryTimer();
            syncIf(profile);
            timings.query += elapsed(start);
        }
        else
        {
            unsigned count = backend_->findContacts(settings);
            endQueryTimer();
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
        if (settings.physics != Physics::Mochi) {
            computeForces();
            auto start = Clock::now();
            kickDrift(particles_.data(), omega_.data(), forces_.data(), torques_.data(),
                      int(particles_.size()), settings.dt, true);
            syncIf(profile);
            timings.integrate += elapsed(start);
            ++steps;
            backend_->waitForCuda();
            updateAccel(rebuildEvery);
            granularForces(true);
            reportBvhCosts();
            start = Clock::now();
            kickDrift(particles_.data(), omega_.data(), forces_.data(), torques_.data(),
                      int(particles_.size()), settings.dt, false);
            syncIf(profile);
            timings.integrate += elapsed(start);
            return;
        }
        auto start = Clock::now();
        integrate(particles_.data(), forces_.data(), oldAccelerations_.data(),
                  int(particles_.size()), settings);
        syncIf(profile);
        timings.integrate += elapsed(start);
        ++steps;
        updateAccel(rebuildEvery);
        computeForces();
        reportBvhCosts();
    }

    void Simulation::updateAccel(int rebuildEvery)
    {
        adaptiveBvh_ = rebuildEvery < 0;
        lastRebuilt_ = adaptiveBvh_ ? bvhOptimizer_.rebuild() : rebuildEvery > 0 && steps % rebuildEvery == 0;
        auto start = Clock::now();
        if (adaptiveBvh_)
            accelGpu_.begin(backend_->defaultStream());
        backend_->update(lastRebuilt_);
        if (adaptiveBvh_)
        {
            accelGpu_.end(backend_->defaultStream());
            lastAccelMs_ = accelGpu_.elapsedTime();
        }
        syncIf(profile);
        timings.accel += elapsed(start);
    }

    void Simulation::beginQueryTimer()
    {
        if (adaptiveBvh_)
            queryGpu_.begin(backend_->launchStream());
    }

    void Simulation::endQueryTimer()
    {
        if (!adaptiveBvh_)
            return;
        queryGpu_.end(backend_->launchStream());
        lastQueryMs_ = queryGpu_.elapsedTime();
    }

    void Simulation::reportBvhCosts()
    {
        if (adaptiveBvh_)
            bvhOptimizer_.update(lastRebuilt_, lastAccelMs_, lastQueryMs_);
    }

    void Simulation::upload(const std::vector<Particle> &particles)
    {
        if (particles.size() != particles_.size())
            throw std::runtime_error("upload() keeps particle IDs and count fixed.");
        particles_.upload(particles);
        forces_.clear();
        oldAccelerations_.clear();
        history_.reset();
        bvhOptimizer_ = bvhOptimizer{};
        forcesReady_ = false;
        if (settings.physics != Physics::Mochi) {
            if (response != Response::Contacts)
                throw std::runtime_error("Granular physics requires --response contacts.");
            omega_.resize(particles.size());
            torques_.resize(particles.size());
            contactFlags_.resize(particles.size());
            wallHistory_.resize(6 * particles.size());
            omega_.clear();
            torques_.clear();
            contactFlags_.clear();
            wallHistory_.clear();
        }
        backend_->waitForCuda();
        backend_->update(true);
        steps = 0;
        timings = {};
        if (settings.physics != Physics::Mochi) {
            computeForces();
            cudaCheck(cudaDeviceSynchronize());
            timings = {};
        }
    }

    std::vector<Contact> Simulation::contacts()
    {
        if (settings.physics != Physics::Mochi) backend_->waitForCuda();
        backend_->findContacts(settings);
        return backend_->contacts();
    }

    OrbitCamera Simulation::defaultCamera() const
    {
        float scale = settings.boxWidth > 0.f ? settings.boxWidth : 1.f;
        Vec3 target = scale * Vec3(0.5f, 0.35f, 0.5f);
        Vec3 eye = scale * Vec3(1.8f, 1.35f, 1.95f);
        if (settings.walls == 4) eye = target + 2.5f * (eye - target);
        return OrbitCamera(eye, target, 0.45f);
    }

    std::vector<uint32_t> Simulation::render(int width, int height, int samples)
    {
        backend_->setView(view, settings.physics == Physics::Mochi ? nullptr : contactFlags_.data());
        if (settings.physics != Physics::Mochi) backend_->waitForCuda();
        return backend_->render(defaultCamera().camera(), width, height, samples);
    }

    void Simulation::renderTo(uint32_t *devicePixels, const Camera &camera,
                              int width, int height, int samples)
    {
        backend_->setView(view, settings.physics == Physics::Mochi ? nullptr : contactFlags_.data());
        if (settings.physics != Physics::Mochi) backend_->waitForCuda();
        backend_->renderTo(camera, devicePixels, width, height, samples);
    }

    void Simulation::granularForces(bool advance) {
        int count = int(particles_.size());
        resetGranular(particles_.data(), forces_.data(), torques_.data(), contactFlags_.data(),
                      count, settings.gravity);
        backend_->waitForCuda();
        auto start = Clock::now();
        beginQueryTimer();
        unsigned pairs = backend_->findContacts(settings);
        endQueryTimer();
        timings.query += elapsed(start);
        start = Clock::now();
        history_.apply(particles_.data(), omega_.data(), forces_.data(), torques_.data(),
                       contactFlags_.data(), backend_->deviceContacts(), pairs, settings, advance);
        applyWalls(particles_.data(), omega_.data(), forces_.data(), torques_.data(),
                   contactFlags_.data(), wallHistory_.data(), count, settings, advance);
        syncIf(profile);
        timings.response += elapsed(start);
        forcesReady_ = true;
    }

    Diagnostics Simulation::diagnostics() {
        Diagnostics result;
        auto state = particles();
        auto angular = omega_.download();
        for (size_t i = 0; i < state.size(); ++i) {
            const auto& p = state[i];
            float speed2 = dot(p.velocity, p.velocity);
            result.maximumSpeed = std::max(result.maximumSpeed, std::sqrt(speed2));
            result.kineticEnergy += 0.5 * p.mass * speed2;
            if (settings.physics != Physics::Mochi)
            {
                result.kineticEnergy += 0.2 * p.mass * p.radius * p.radius * dot(angular[i], angular[i]);
                result.maximumAngularSpeed = std::max(result.maximumAngularSpeed, length(angular[i]));
            }
        }
        auto addState = [&](const ContactState& contact) {
            if (contact.normalForce > 0.f) {
                ++result.contacts;
                result.sliding += contact.sliding;
                result.rollingYielding += contact.rollingYielding;
            }
        };
        if (settings.physics == Physics::Mochi) result.contacts = unsigned(contacts().size());
        else {
            for (auto contact : contactStates()) addState(contact.state);
            for (auto contact : wallStates()) addState(contact);
            result.historyBytes = history_.bytes() + wallHistory_.size() * sizeof(ContactState);
        }
        return result;
    }

    void Simulation::saveGranularState(const std::string& path) const {
        if (settings.physics == Physics::Mochi) return;
        auto angular = angularVelocities();
        auto torque = torques();
        std::ofstream rotation(path + ".rotation.csv");
        if (!rotation) throw std::runtime_error("Could not write angular state: " + path);
        rotation << "id,wx,wy,wz,tx,ty,tz\n" << std::setprecision(9);
        for (size_t i = 0; i < angular.size(); ++i)
            rotation << i << ',' << angular[i].x << ',' << angular[i].y << ',' << angular[i].z
                     << ',' << torque[i].x << ',' << torque[i].y << ',' << torque[i].z << '\n';
        auto writeContact = [](std::ostream& out, unsigned a, unsigned b, const ContactState& c) {
            out << a << ',' << b << ',' << c.displacement.x << ',' << c.displacement.y << ','
                << c.displacement.z << ',' << c.tangentialForce.x << ',' << c.tangentialForce.y
                << ',' << c.tangentialForce.z << ',' << c.normalForce << ',' << c.sliding
                << ',' << c.rollingDisplacement.x << ',' << c.rollingDisplacement.y << ',' << c.rollingDisplacement.z
                << ',' << c.rollingTorque.x << ',' << c.rollingTorque.y << ',' << c.rollingTorque.z
                << ',' << c.rollingYielding << '\n';
        };
        std::ofstream pairs(path + ".contacts.csv"), walls(path + ".walls.csv");
        if (!pairs || !walls) throw std::runtime_error("Could not write contact state: " + path);
        const char* header = "a,b,sx,sy,sz,ftx,fty,ftz,fn,sliding,rsx,rsy,rsz,rtx,rty,rtz,rolling_yielding\n";
        pairs << header << std::setprecision(9);
        walls << header << std::setprecision(9);
        for (auto contact : contactStates())
            writeContact(pairs, unsigned(contact.key >> 32), unsigned(contact.key), contact.state);
        auto wall = wallStates();
        for (size_t i = 0; i < wall.size(); ++i)
            if (wall[i].normalForce > 0.f) writeContact(walls, unsigned(i / 6), unsigned(i % 6), wall[i]);
    }

} // namespace oops
