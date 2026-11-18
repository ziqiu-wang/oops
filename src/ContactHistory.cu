#include "ContactHistory.h"
#include "ContactModel.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm>

namespace oops {
namespace {

__device__ void add(Vec3* destination, Vec3 value) {
    atomicAdd(&destination->x, value.x);
    atomicAdd(&destination->y, value.y);
    atomicAdd(&destination->z, value.z);
}

__global__ void makeKeys(const Contact* contacts, uint64_t* keys, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) keys[i] = contactKey(contacts[i].a, contacts[i].b);
}

__global__ void response(const Particle* particles, const Vec3* omega, Vec3* forces, Vec3* torques,
    unsigned* flags, const uint64_t* keys, ContactState* next, unsigned count, const uint64_t* oldKeys,
    const ContactState* previous, unsigned oldCount, Settings settings,
    ContactParameters parameters, float dt) {
    unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    uint64_t key = keys[slot];
    unsigned lo = 0, hi = oldCount;
    while (lo < hi) {
        unsigned mid = lo + (hi - lo) / 2;
        if (oldKeys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    ContactState old;
    if (lo < oldCount && oldKeys[lo] == key) old = previous[lo];
    unsigned a = unsigned(key >> 32), b = unsigned(key);
    auto result = sphereContact(particles[a], particles[b], omega[a], omega[b],
                                old, settings, parameters, dt);
    next[slot] = result.state;
    add(forces + a, result.force);
    add(forces + b, -result.force);
    add(torques + a, result.torqueA);
    add(torques + b, result.torqueB);
    if (result.state.normalForce > 0.f) {
        atomicMax(flags + a, 1u + result.state.sliding);
        atomicMax(flags + b, 1u + result.state.sliding);
    }
}

__global__ void resetKernel(const Particle* particles, Vec3* forces, Vec3* torques, unsigned* flags,
                             int count, Vec3 gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        forces[i] = particles[i].mass * gravity;
        torques[i] = Vec3(0.f);
        flags[i] = 0;
    }
}

__global__ void kickKernel(Particle* particles, Vec3* omega, const Vec3* forces,
                            const Vec3* torques, int count, float dt, bool drift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Particle& p = particles[i];
    p.velocity += (0.5f * dt / p.mass) * forces[i];
    omega[i] += (0.5f * dt * inverseInertia(p)) * torques[i];
    if (drift) p.position += dt * p.velocity;
}

__global__ void wallKernel(const Particle* particles, const Vec3* omega, Vec3* forces,
    Vec3* torques, unsigned* flags, ContactState* history, int count, Settings settings,
    ContactParameters parameters, bool advance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // Particle contacts finish first on this stream; each thread owns its six walls.
    for (int wall = 0; wall < 6; ++wall) {
        auto result = wallContact(particles[i], omega[i], wall, history[6 * i + wall],
                                   settings, parameters, advance ? settings.dt : 0.f);
        forces[i] += result.force;
        torques[i] += result.torqueA;
        if (result.state.normalForce > 0.f)
            flags[i] = max(flags[i], 1u + result.state.sliding);
        if (advance) history[6 * i + wall] = result.state;
    }
}
}

void ContactHistory::apply(const Particle* particles, const Vec3* omega, Vec3* forces, Vec3* torques,
    unsigned* flags, const Contact* contacts, unsigned count, const Settings& settings, bool advance) {
    // Grow only scratch storage. The committed previous history remains untouched.
    if (count > nextKeys_.size()) {
        size_t capacity = std::max<size_t>(count, 2 * nextKeys_.size());
        nextKeys_.resize(capacity);
        next_.resize(capacity);
    }
    if (count) {
        makeKeys<<<(count + 255) / 256, 256>>>(contacts, nextKeys_.data(), count);
        auto keys = thrust::device_pointer_cast(nextKeys_.data());
        thrust::sort(keys, keys + count);
        response<<<(count + 255) / 256, 256>>>(particles, omega, forces, torques,
            flags, nextKeys_.data(), next_.data(), count, previousKeys_.data(), previous_.data(),
            count_, settings, contactParameters(settings), advance ? settings.dt : 0.f);
        cudaCheck(cudaGetLastError());
    }
    viewCount_ = count;
    viewPrevious_ = advance;
    if (advance) {
        previousKeys_.swap(nextKeys_);
        previous_.swap(next_);
        count_ = count;
    }
}

std::vector<StoredContact> ContactHistory::download() const {
    auto keys = (viewPrevious_ ? previousKeys_ : nextKeys_).download(viewCount_);
    auto states = (viewPrevious_ ? previous_ : next_).download(viewCount_);
    std::vector<StoredContact> result(viewCount_);
    for (unsigned i = 0; i < viewCount_; ++i) result[i] = {keys[i], states[i]};
    return result;
}

size_t ContactHistory::bytes() const {
    return (previousKeys_.size() + nextKeys_.size()) * sizeof(uint64_t)
         + (previous_.size() + next_.size()) * sizeof(ContactState);
}

void resetGranular(const Particle* particles, Vec3* forces, Vec3* torques, unsigned* flags,
                   int count, Vec3 gravity) {
    resetKernel<<<(count + 255) / 256, 256>>>(particles, forces, torques, flags, count, gravity);
    cudaCheck(cudaGetLastError());
}

void kickDrift(Particle* particles, Vec3* omega, const Vec3* forces, const Vec3* torques,
               int count, float dt, bool drift) {
    kickKernel<<<(count + 255) / 256, 256>>>(particles, omega, forces, torques, count, dt, drift);
    cudaCheck(cudaGetLastError());
}

void applyWalls(const Particle* particles, const Vec3* omega, Vec3* forces, Vec3* torques,
                unsigned* flags, ContactState* history, int count, const Settings& settings, bool advance) {
    wallKernel<<<(count + 255) / 256, 256>>>(particles, omega, forces, torques, flags, history,
                                         count, settings, contactParameters(settings), advance);
    cudaCheck(cudaGetLastError());
}

} // namespace oops
