#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include "Window.h"
#include <string>

namespace oops {

namespace {
const char* vertexSource = R"(
#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // (0,0) (2,0) (0,2)
    uv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* fragmentSource = R"(
#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D image;
void main() { color = texture(image, uv); }
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
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
    const char* title):
width_(width),
height_(height) {
    if (!glfwInit()) {
        throw std::runtime_error("glfwInit failed in window.cpp");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) { 
        glfwTerminate(); 
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window_);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        throw std::runtime_error("glewInit failed");
    }

    glfwSwapInterval(0);
    glfwSetWindowUserPointer(window_, this);
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double dy) {
        static_cast<Window*>(glfwGetWindowUserPointer(w))->scrollAccum_ += float(dy);
    });
    glfwGetCursorPos(window_, &lastX_, &lastY_);


    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_t(width) * height * 4, nullptr, GL_DYNAMIC_DRAW); // opengl buffer for rgba image
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (cudaGraphicsGLRegisterBuffer(&resource_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard)
        != cudaSuccess) {
        glDeleteBuffers(1, &pbo_);
        glfwTerminate();
        throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed");
    }

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

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
    if (!linked) throw std::runtime_error("shader program link failed");

    glGenVertexArrays(1, &vao_);
}

Window::~Window() {
    if (resource_) cudaGraphicsUnregisterResource(resource_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);
    glDeleteTextures(1, &texture_);
    glDeleteBuffers(1, &pbo_);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

uint32_t* Window::map() {
    cudaGraphicsMapResources(1, &resource_);
    void* ptr = nullptr;
    size_t size = 0;
    cudaGraphicsResourceGetMappedPointer(&ptr, &size, resource_);
    return static_cast<uint32_t*>(ptr);
}

void Window::unmap() {
    cudaGraphicsUnmapResources(1, &resource_);
}

void Window::present() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window_);
    glfwPollEvents();

    input_.orbiting = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    input_.panning  = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    double x, y;
    glfwGetCursorPos(window_, &x, &y);
    input_.mouseDx = float(x - lastX_);
    input_.mouseDy = float(y - lastY_);
    lastX_ = x;
    lastY_ = y;

    input_.scroll = scrollAccum_;
    scrollAccum_ = 0.f;

}

}