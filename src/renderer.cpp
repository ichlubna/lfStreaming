#include <GL/glew.h>
#include <vector>
#include <stdexcept>
#include "renderer.h"

const char *vertexShaderSource = R""""( 
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 inTexCoord;
out vec2 texCoord;
void main(){
gl_Position = vec4(position, 1.0);
texCoord = inTexCoord;})"""";

const char *fragmentShaderSource = R""""(
#version 330 core
in vec2 texCoord;
out vec4 color;
uniform sampler2D image;
void main(){
color = texture(image, texCoord);})"""";

Renderer::Renderer()
{
}

void Renderer::createWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(initialResolution.x, initialResolution.y, windowName, NULL, NULL);
    if(!window)
        throw std::runtime_error("Cannot create window");
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, []([[maybe_unused]]GLFWwindow * window, int width, int height)
    {
          glViewport(0, 0, width, height);
    });
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED | GLFW_CURSOR_HIDDEN);
    glfwSetWindowUserPointer(window, this);
    glfwSetCursorPosCallback(window, [](GLFWwindow * window, double xPos, double yPos)
    {
        auto *renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
        renderer->setMousePosition({xPos, yPos});
    });
     glfwSetKeyCallback(window, []([[maybe_unused]]GLFWwindow * window, int key, [[maybe_unused]]int scancode, [[maybe_unused]]int action, [[maybe_unused]]int mods)
    {
        auto *renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
        switch(key)
        {
        case GLFW_KEY_ESCAPE:
            renderer->quit();
            break;

        default:
            break;
        }
    });
}

void Renderer::loadShaders()
{
    glm::uvec2 shaderIDs{glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER)};
    glShaderSource(shaderIDs.x, 1, &vertexShaderSource, NULL);
    glShaderSource(shaderIDs.y, 1, &fragmentShaderSource, NULL);
    glCompileShader(shaderIDs.x);
    glCompileShader(shaderIDs.y);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, shaderIDs.x);
    glAttachShader(shaderProgram, shaderIDs.y);
    glLinkProgram(shaderProgram);
    glDeleteShader(shaderIDs.x);
    glDeleteShader(shaderIDs.y);
}

void Renderer::prepareQuad()
{
    constexpr size_t VERTEX_SIZE{5*sizeof(GLfloat)};
    constexpr size_t UV_OFFSET{3*sizeof(GLfloat)};
    
    std::vector<float> vertices{
    -1.0f, 3.0f, 0.0f,  0.0f, 2.0f,
    3.0f, -1.0f, 0.0f,   2.0f, 0.0f,
    -1.0f,-1.0f, 0.0f,  0.0f, 0.0f};


    unsigned int vbo, vao;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE, (char*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, VERTEX_SIZE, (char*)0 + UV_OFFSET);
    glBindBuffer(GL_ARRAY_BUFFER, 0);  
}
unsigned int Renderer::getTexture(glm::ivec2 resolution)
{
    glDeleteTextures(1, &texture);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resolution.x, resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    return texture;
}

void Renderer::setupGL()
{
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("Cannot init glew");

    loadShaders();
    prepareQuad();
}

void Renderer::init()
{
    createWindow();
    setupGL(); 
    prepared = true;
}

void Renderer::render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwSwapBuffers(window);
    glfwPollEvents();
}
