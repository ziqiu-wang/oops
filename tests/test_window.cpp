#include "Window.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>

int main()
{
    try
    {
        oops::Window window(800, 600, "window test");
        while (!window.shouldClose())
        {
            uint32_t *pixels = window.map();
            cudaMemset(pixels, 0x80, 800 * 600 * 4);
            cudaMemset(pixels, 0xFF, 800 * 4);
            window.unmap();
            window.present();

            const auto &in = window.input();
            if (in.orbiting || in.panning || in.scroll != 0.f)
                printf("orbit=%d pan=%d dx=%.0f dy=%.0f scroll=%.1f\n",
                       in.orbiting, in.panning, in.mouseDx, in.mouseDy, in.scroll);
        }
    }
    catch (const std::exception &e)
    {
        printf("error: %s\n", e.what());
        return 1;
    }
}
