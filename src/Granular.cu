#include "Granular.h"
#include "CudaBuffer.h"

namespace oops {

__global__ void resetKernel(Particle* particles, Vec3* forces, int count, Vec3 gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) forces[i] = particles[i].mass * gravity;
}

__device__ void addForce(Vec3* destination, Vec3 force) {
    atomicAdd(&destination->x, force.x);
    atomicAdd(&destination->y, force.y);
    atomicAdd(&destination->z, force.z);
}

__global__ void responseKernel(Particle* particles, Vec3* forces, const Contact* contacts,
                               int count, ContactParameters parameters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Contact c = contacts[i];
    Vec3 force = contactForce(particles[c.a], particles[c.b], parameters);
    addForce(forces + c.a, force);
    addForce(forces + c.b, -force);
}

__global__ void integrateKernel(Particle* particles, Vec3* forces, Vec3* previous,
                                int count, Settings settings) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) integrateParticle(particles[i], forces[i], previous[i], settings);
}

void resetForces(Particle* particles, Vec3* forces, int count, Vec3 gravity) {
    resetKernel<<<(count + 255) / 256, 256>>>(particles, forces, count, gravity);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void applyContacts(Particle* particles, Vec3* forces, const Contact* contacts,
                   int count, const Settings& settings) {
    if (count == 0) return;
    responseKernel<<<(count + 255) / 256, 256>>>(particles, forces, contacts, count,
                                               contactParameters(settings));
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void integrate(Particle* particles, Vec3* forces, Vec3* previous,
               int count, const Settings& settings) {
    integrateKernel<<<(count + 255) / 256, 256>>>(particles, forces, previous, count, settings);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace oops
