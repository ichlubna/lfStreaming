#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Renderer
{
    public:
    Renderer();
    void render();
    void init();
    void inputs();
    void setMousePosition(glm::vec2 position){mousePosition=position;};
    bool ready(){return prepared;};

    private:
    bool prepared{false};
    unsigned int shaderProgram;
    unsigned int texture;
    glm::uvec2 initialResolution{1280, 720};
    const char *windowName{"Lightfield"};
    glm::vec2 mousePosition;
    GLFWwindow *window;
    void createWindow();
    void loadShaders();
    void prepareQuad();
    void generateTexture();
    void setupGL();
    void quit(){prepared=false;};
};
