#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Renderer
{
    public:
        Renderer();
        void render();
        void init();
        void inputs();
        void setMousePosition(glm::vec2 position)
        {
            mousePosition = {position.y, position.x};
            mouseMoved = true;
        };
        glm::vec2 getMousePosition()
        {
            return mousePosition;
        }
        bool mouseChanged()
        {
            bool move = mouseMoved;
            mouseMoved = false;
            return move;
        }

        bool ready()
        {
            return prepared;
        };

        bool isPaused()
        {
            return paused;
        };
        unsigned int getTexture(glm::ivec2 resolution);

    private:
        bool prepared{false};
        bool mouseMoved{true};
        bool paused{false};
        unsigned int shaderProgram;
        unsigned int texture;
        glm::uvec2 initialResolution{1920, 1080};
        const char *windowName{"Lightfield"};
        glm::vec2 mousePosition;
        GLFWwindow *window;
        void createWindow();
        void loadShaders();
        void prepareQuad();
        void setupGL();
        void quit()
        {
            prepared = false;
        };
        void pause()
        {
            paused = !paused;
        };
};
