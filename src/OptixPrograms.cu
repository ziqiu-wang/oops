#include "GpuData.h"
#include "Granular.h"
#include <owl/owl_device.h>
#include <optix_device.h>

using namespace oops;

DECLARE_OPTIX_LAUNCH_PARAMS(LaunchData)

OPTIX_BOUNDS_PROGRAM(proxyBounds)(const void* data, owl::box3f& bounds, int id) {
    Particle p = static_cast<const GeometryData*>(data)->particles[id];
    Vec3 lower = p.position - 2.f * p.radius;
    Vec3 upper = p.position + 2.f * p.radius;
    for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = nextafterf(lower[axis], -CUDART_INF_F);
        upper[axis] = nextafterf(upper[axis], CUDART_INF_F);
    }
    bounds = owl::box3f(lower, upper);
}

__device__ void addForce(Vec3* destination, Vec3 force) {
    atomicAdd(&destination->x, force.x);
    atomicAdd(&destination->y, force.y);
    atomicAdd(&destination->z, force.z);
}

OPTIX_INTERSECT_PROGRAM(collision)() {
    const auto& data = optixLaunchParams;
    unsigned i = optixGetLaunchIndex().x;
    unsigned j = optixGetPrimitiveIndex();
    if (i == j) return;
    const Particle& a = data.particles[i];
    const Particle& b = data.particles[j];
    Vec3 difference = b.position - a.position;
    float d2 = dot(difference, difference);

    if (data.settings.strictUpstream) {
        float distance = sqrtf(d2);
        if (!(2.f * b.radius - distance > 0.f
              && a.radius + b.radius - distance > 0.f
              && (i < j || 2.f * a.radius - distance < 0.f))) return;
    } else {
        float sum = a.radius + b.radius;
        if (d2 > sum * sum || d2 > 4.f * b.radius * b.radius) return;
        bool reverseSeesPair = d2 <= 4.f * a.radius * a.radius;
        if (reverseSeesPair && i > j) return;
    }

    if (data.emitContacts) {
        unsigned slot = atomicAdd(data.contactCount, 1u);
        if (slot < data.capacity)
            data.contacts[slot] = i < j ? Contact{i, j} : Contact{j, i};
    } else {
        Vec3 force = contactForce(a, b, data.contact);
        addForce(data.forces + i, force);
        addForce(data.forces + j, -force);
    }
}

OPTIX_RAYGEN_PROGRAM(query)() {
    unsigned i = optixGetLaunchIndex().x;
    unsigned payload = i;
    owl::RayT<0, 2> ray(optixLaunchParams.particles[i].position,
                        Vec3(0.f, 0.f, 1.f), 0.f, 1.e-16f);
    owl::traceRay(optixLaunchParams.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
}

struct SurfaceHit {
    Vec3 position, normal, albedo;
    bool found = false;
};

OPTIX_INTERSECT_PROGRAM(sphere)() {
    unsigned id = optixGetPrimitiveIndex();
    const Particle& p = optixLaunchParams.particles[id];
    Vec3 origin = optixGetObjectRayOrigin();
    Vec3 direction = optixGetObjectRayDirection();
    Vec3 offset = origin - p.position;
    float a = dot(direction, direction);
    float b = dot(offset, direction);
    float c = dot(offset, offset) - p.radius * p.radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.f) return;
    float root = sqrtf(discriminant);
    float t = (-b - root) / a;
    if (t <= optixGetRayTmin()) t = (-b + root) / a;
    if (t > optixGetRayTmin() && t < optixGetRayTmax()) optixReportIntersection(t, 0);
}

OPTIX_CLOSEST_HIT_PROGRAM(surface)() {
    auto& hit = owl::getPRD<SurfaceHit>();
    unsigned id = optixGetPrimitiveIndex();
    const Particle& p = optixLaunchParams.particles[id];
    Vec3 direction = optixGetWorldRayDirection();
    hit.position = Vec3(optixGetWorldRayOrigin()) + optixGetRayTmax() * direction;
    hit.normal = normalize(hit.position - p.position);
    if (dot(hit.normal, direction) > 0.f) hit.normal = -hit.normal;
    float shade = float((id * 2654435761u) >> 24) / 255.f;
    hit.albedo = Vec3(0.65f + 0.2f * shade, 0.38f + 0.2f * shade, 0.16f + 0.18f * shade);
    hit.found = true;
}

OPTIX_MISS_PROGRAM(sky)() { owl::getPRD<SurfaceHit>().found = false; }

__device__ float randomFloat(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return float(state >> 8) * (1.f / 16777216.f);
}

__device__ Vec3 randomUnitVector(unsigned& state) {
    float z = 2.f * randomFloat(state) - 1.f;
    float phi = 6.283185307f * randomFloat(state);
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    return Vec3(r * cosf(phi), z, r * sinf(phi));
}

OPTIX_RAYGEN_PROGRAM(image)() {
    const auto& data = optixLaunchParams;
    auto index = optixGetLaunchIndex();
    auto size = optixGetLaunchDimensions();
    unsigned state = 1u + index.x + size.x * index.y;
    Vec3 color(0.f);
    for (int sample = 0; sample < data.samples; ++sample) {
        float x = (2.f * (index.x + randomFloat(state)) / size.x - 1.f) * float(size.x) / size.y;
        float y = 1.f - 2.f * (index.y + randomFloat(state)) / size.y;
        Vec3 origin = data.camera.origin;
        Vec3 direction = normalize(data.camera.forward + x * data.camera.right + y * data.camera.up);
        Vec3 throughput(1.f);
        for (int bounce = 0; bounce < 4; ++bounce) {
            SurfaceHit hit;
            owl::RayT<1, 2> ray(origin, direction, 1.e-5f, 1.e20f);
            owl::traceRay(data.world, ray, hit);
            if (!hit.found) {
                float t = 0.5f * (direction.y + 1.f);
                color += throughput * ((1.f - t) * Vec3(0.9f) + t * Vec3(0.45f, 0.65f, 0.95f));
                break;
            }
            throughput *= hit.albedo;
            origin = hit.position + 1.e-5f * hit.normal;
            direction = normalize(hit.normal + randomUnitVector(state));
        }
    }
    color /= float(data.samples);
    color = Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));
    data.pixels[index.x + size.x * index.y] = owl::make_rgba(color);
}
