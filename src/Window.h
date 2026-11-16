#pragma once
#include <cstdint>

struct GLFWwindow;
struct cudaGraphicsResource;

namespace oops
{

    struct Input
    {
        bool orbiting = false;
        bool panning = false;
        float mouseDx = 0.f;
        float mouseDy = 0.f;
        float scroll = 0.f;
    };

    class Window
    {
    public:
        Window(int width, int height, const char *title);
        ~Window();
        // ban copying
        Window(const Window &) = delete;
        Window &operator=(const Window &) = delete;

        bool shouldClose() const;
        uint32_t *map();
        void unmap();
        void present();

        const Input &input() const
        {
            return input_;
        }

        int width() const { return width_; }
        int height() const { return height_; }
        bool minimized() const { return width_ == 0 || height_ == 0; }

    private:
        int width_, height_;
        GLFWwindow *window_ = nullptr;
        unsigned pbo_ = 0, texture_ = 0, vao_ = 0, program_ = 0;
        cudaGraphicsResource *resource_ = nullptr;

        Input input_;
        double lastX_ = 0.0, lastY_ = 0.0;
        float scrollAccum_ = 0.f;
        void readInput();

        void createBuffers(int width, int height);
        void destroyBuffers();
        int pendingWidth_ = 0, pendingHeight_ = 0;
        bool resized_ = false;
    };

}