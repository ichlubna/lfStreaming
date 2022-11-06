#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Renderer
{
    public:
    Renderer();
    void render();
    void inputs();
    void setMousePosition(glm::vec2 position){mousePosition=position;};
    bool endSignaled(){return endSignal;};

    private:
    bool endSignal{false};
    unsigned int shaderProgram;
    glm::uvec2 initialResolution{1280, 720};
    const char *windowName{"Lightfield"};
    glm::vec2 mousePosition;
    GLFWwindow *window;
    void createWindow();
    void loadShaders();
    void prepareQuad();
    void setupGL();
    void init();
    void quit(){endSignal=true;};
};
