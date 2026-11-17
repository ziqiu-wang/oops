#pragma once
#include "GpuData.h"
#include "Window.h"
#include <algorithm>
#include <cmath>

namespace oops
{

    inline Camera lookAt(Vec3 eye, Vec3 target, float halfFov)
    {
        Vec3 forward = normalize(target - eye);
        Vec3 right = normalize(cross(forward, Vec3(0.f, 1.f, 0.f)));
        Vec3 up = cross(right, forward);
        return {eye, forward, right * halfFov, up * halfFov};
    }

    class OrbitCamera
    {
    public:
        OrbitCamera(Vec3 eye, Vec3 target, float halfFov = 0.45f)
            : target_(target), halfFov_(halfFov)
        {
            Vec3 offset = eye - target;
            distance_ = length(offset);
            pitch_ = std::asin(offset.y / distance_);
            yaw_ = std::atan2(offset.z, offset.x);
        }

        void update(const Input &in)
        {
            if (in.orbiting)
            {
                yaw_ += in.mouseDx * orbitSpeed;
                pitch_ += in.mouseDy * orbitSpeed;
                pitch_ = std::clamp(pitch_, -1.5f, 1.5f);
            }
            if (in.panning)
            {
                Camera c = camera();
                float scale = distance_ * panSpeed;
                target_ -= normalize(c.right) * (in.mouseDx * scale);
                target_ += normalize(c.up) * (in.mouseDy * scale);
            }
            if (in.scroll != 0.f)
            {
                distance_ *= std::pow(zoomFactor, in.scroll);
                distance_ = std::clamp(distance_, 0.05f, 20.f);
            }
        }

        Vec3 eye() const
        {
            return target_ + distance_ * Vec3(std::cos(pitch_) * std::cos(yaw_), std::sin(pitch_), std::cos(pitch_) * std::sin(yaw_));
        }

        Camera camera() const { return lookAt(eye(), target_, halfFov_); }

        float orbitSpeed = 0.005f;
        float panSpeed = 0.0015f;
        float zoomFactor = 0.9f;

    private:
        Vec3 target_;
        float distance_, yaw_, pitch_, halfFov_;
    };

}
