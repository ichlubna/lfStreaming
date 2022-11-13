#include <string>
#include <vector>
#include <memory>
#include "renderer.h"

class Decoder
{
    public:
    Decoder(std::string input);
    void decodeAndPlay();
    void decodeAndStore(std::string trajectory);

    private:
    class SelectedFrames
    {
        public:
        class Frame
        {
            public:
            glm::ivec2 coord;
            float weight;
        };
        Frame topLeft, topRight, bottomLeft, bottomRight;
        float topInter() const {return topRight.weight/(topLeft.weight+topRight.weight);}
        float bottomInter() const {return bottomRight.weight/(bottomLeft.weight+bottomRight.weight);} 
        glm::vec3 topBottomMiddleInter() const { float top = topInter(); float bottom = bottomInter(); return {top, bottom, bottom/(top+bottom)};};
    };

    std::string input;
    std::unique_ptr<Renderer> renderer;
    void init();
    void initCodec();
    std::vector<glm::vec2> parseTrajectory(std::string textTrajectory) const;
    SelectedFrames pickFromGrid(glm::uvec2 gridSize, glm::vec2 position) const;
    void storeImage(std::vector<uint8_t> *data, glm::uvec2 resolution, std::string path);
};
