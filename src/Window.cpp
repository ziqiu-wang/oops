#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include <string>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "Window.h"

namespace oops
{

    namespace
    {
        const char *vertexSource = R"(
#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // (0,0) (2,0) (0,2)
    uv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

        const char *fragmentSource = R"(
#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D image;
void main() { color = texture(image, uv); }
)";

        GLuint compileShader(GLenum type, const char *source)
        {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint ok = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                char log[1024];
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                glDeleteShader(shader);
                throw std::runtime_error(std::string("shader compile failed: ") + log);
            }
            return shader;
        }
    }

    Window::Window(
        int width,
        int height,
        const char *title) : width_(width),
                             height_(height)
    {
        if (!glfwInit())
        {
            throw std::runtime_error("glfwInit failed in window.cpp");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_)
        {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow failed");
        }
        glfwMakeContextCurrent(window_);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK)
        {
            glfwTerminate();
            throw std::runtime_error("glewInit failed");
        }

        glfwSwapInterval(0);
        glfwSetWindowUserPointer(window_, this);
        glfwSetScrollCallback(window_, [](GLFWwindow *w, double, double dy)
                              { static_cast<Window *>(glfwGetWindowUserPointer(w))->scrollAccum_ += float(dy); });
        glfwGetCursorPos(window_, &lastX_, &lastY_);

        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow *w, int width, int height)
                                       {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        self->pendingWidth_ = width;
        self->pendingHeight_ = height;
        self->resized_ = true; });

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
        createBuffers(fbWidth, fbHeight);

        GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSource);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint linked = 0;
        glGetProgramiv(program_, GL_LINK_STATUS, &linked);
        if (!linked)
            throw std::runtime_error("shader program link failed");

        glGenVertexArrays(1, &vao_);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");
    }

    Window::~Window()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        destroyBuffers();
        glDeleteVertexArrays(1, &vao_);
        glDeleteProgram(program_);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void Window::createBuffers(int width, int height)
    {
        width_ = width;
        height_ = height;
        if (width == 0 || height == 0)
            return;

        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_t(width) * height * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        if (cudaGraphicsGLRegisterBuffer(&resource_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess)
            throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed");

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void Window::destroyBuffers()
    {
        if (resource_)
            cudaGraphicsUnregisterResource(resource_);
        if (pbo_)
            glDeleteBuffers(1, &pbo_);
        if (texture_)
            glDeleteTextures(1, &texture_);
        resource_ = nullptr;
        pbo_ = texture_ = 0;
    }

    bool Window::shouldClose() const
    {
        return glfwWindowShouldClose(window_);
    }

    uint32_t *Window::map()
    {
        if (resized_)
        {
            resized_ = false;
            if (pendingWidth_ != width_ || pendingHeight_ != height_)
            {
                destroyBuffers();
                createBuffers(pendingWidth_, pendingHeight_);
            }
        }
        if (minimized())
        {
            return nullptr;
        }

        cudaGraphicsMapResources(1, &resource_);
        void *ptr = nullptr;
        size_t size = 0;
        cudaGraphicsResourceGetMappedPointer(&ptr, &size, resource_);
        return static_cast<uint32_t *>(ptr);
    }

    void Window::unmap()
    {
        if (!resource_)
        {
            return;
        }
        cudaGraphicsUnmapResources(1, &resource_);
    }

    void Window::readInput()
    {
        glfwPollEvents();

        input_.orbiting = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        input_.panning = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        double x, y;
        glfwGetCursorPos(window_, &x, &y);
        input_.mouseDx = float(x - lastX_);
        input_.mouseDy = float(y - lastY_);
        lastX_ = x;
        lastY_ = y;

        input_.scroll = scrollAccum_;
        scrollAccum_ = 0.f;

        if (ImGui::GetIO().WantCaptureMouse)
        {
            input_.orbiting = false;
            input_.panning = false;
            input_.scroll = 0.f;
        }
    }

    void Window::present()
    {
        if (minimized())
        {
            readInput();
            ImGui::Render();
            return;
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glUseProgram(program_);
        glBindVertexArray(vao_);
        glViewport(0, 0, width_, height_);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);
        readInput();
    }

    void Window::beginUi()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

}