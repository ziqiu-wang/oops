// for testing glfw and glew build and link

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>

int main() {
    if (!glfwInit()) {
        printf("glfwInit failed\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "smoke", nullptr, nullptr);
    if (!window) {
        printf("glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        printf("glewInit failed\n");
        glfwTerminate();
        return 1;
    }

    printf("Renderer: %s\n", (const char*)glGetString(GL_RENDERER));
    printf("OpenGL:   %s\n", (const char*)glGetString(GL_VERSION));

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
