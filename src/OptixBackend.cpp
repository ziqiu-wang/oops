#include "OptixBackend.h"
#include "Granular.h"
#include <cuda.h>
#include <owl/owl.h>
#include <algorithm>

extern "C" char OptixPrograms_ptx[];

namespace oops {

struct OptixBackend::Impl {
    OWLContext context;
    OWLGroup geometryGroup, world;
    OWLRayGen query, image;
    OWLParams params;

    Impl(Particle* particles, int count) {
        context = owlContextCreate(nullptr, 1);
        owlContextSetRayTypeCount(context, 2);
        OWLModule module = owlModuleCreate(context, OptixPrograms_ptx);
        OWLVarDecl geometryVars[] = {
            {"particles", OWL_RAW_POINTER, OWL_OFFSETOF(GeometryData, particles)},
            {nullptr}
        };
        OWLGeomType type = owlGeomTypeCreate(context, OWL_GEOMETRY_USER,
                                             sizeof(GeometryData), geometryVars, -1);
        owlGeomTypeSetBoundsProg(type, module, "proxyBounds");
        owlGeomTypeSetIntersectProg(type, 0, module, "collision");
        owlGeomTypeSetIntersectProg(type, 1, module, "sphere");
        owlGeomTypeSetClosestHit(type, 1, module, "surface");
        OWLVarDecl noVars[] = {{nullptr}};
        query = owlRayGenCreate(context, module, "query", 0, noVars, -1);
        image = owlRayGenCreate(context, module, "image", 0, noVars, -1);
        OWLMissProg miss = owlMissProgCreate(context, module, "sky", 0, noVars, -1);
        owlMissProgSet(context, 1, miss);
        OWLVarDecl paramsVars[] = {{"data", OWL_USER_TYPE(LaunchData), 0}, {nullptr}};
        params = owlParamsCreate(context, sizeof(LaunchData), paramsVars, -1);
        owlBuildPrograms(context);

        OWLGeom geometry = owlGeomCreate(context, type);
        owlGeomSetPrimCount(geometry, count);
        owlGeomSetPointer(geometry, "particles", particles);
        unsigned flags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry, flags);
        owlGroupBuildAccel(geometryGroup);
        world = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr,
                                       OWL_MATRIX_FORMAT_OWL, flags);
        owlGroupBuildAccel(world);
        owlBuildPipeline(context);
        owlBuildSBT(context);
    }
    ~Impl() { owlContextDestroy(context); }

    void run(OWLRayGen program, const LaunchData& data, int width, int height) {
        owlParamsSetRaw(params, "data", &data);
        owlLaunch2D(program, width, height, params);
    }
};

OptixBackend::OptixBackend(Particle* particles, int count, unsigned capacity)
    : impl_(std::make_unique<Impl>(particles, count)), particleCount_(count),
      initialCapacity_(capacity ? capacity : std::max(1, count * 16)) {
    launch_.particles = particles;
    launch_.world = owlGroupGetTraversable(impl_->world, 0);
}

OptixBackend::~OptixBackend() = default;

void OptixBackend::update(bool rebuild) {
    if (rebuild) {
        owlGroupBuildAccel(impl_->geometryGroup);
        owlGroupBuildAccel(impl_->world);
    } else {
        owlGroupRefitAccel(impl_->geometryGroup);
        owlGroupRefitAccel(impl_->world);
    }
    launch_.world = owlGroupGetTraversable(impl_->world, 0);
}

unsigned OptixBackend::findContacts(const Settings& settings) {
    if (contacts_.size() == 0) contacts_.resize(initialCapacity_);
    launch_.settings = settings;
    launch_.emitContacts = true;
    launch_.contactCount = counter_.data();
    for (;;) {
        counter_.clear();
        launch_.contacts = contacts_.data();
        launch_.capacity = static_cast<unsigned>(contacts_.size());
        impl_->run(impl_->query, launch_, particleCount_, 1);
        contactCount_ = counter_.download()[0];
        if (contactCount_ <= contacts_.size()) return contactCount_;
        contacts_.resize(std::max<size_t>(contactCount_, contacts_.size() * 2));
    }
}

void OptixBackend::accumulateForces(Vec3* forces, const Settings& settings) {
    launch_.settings = settings;
    launch_.contact = contactParameters(settings);
    launch_.emitContacts = false;
    launch_.forces = forces;
    impl_->run(impl_->query, launch_, particleCount_, 1);
}

std::vector<Contact> OptixBackend::contacts() const {
    auto result = contacts_.download(contactCount_);
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<uint32_t> OptixBackend::render(const Camera& camera, int width,
                                          int height, int samples) {
    if (pixels_.size() != size_t(width) * height) pixels_.resize(size_t(width) * height);
    launch_.pixels = pixels_.data();
    launch_.camera = camera;
    launch_.samples = samples;
    impl_->run(impl_->image, launch_, width, height);
    return pixels_.download();
}

} // namespace oops
