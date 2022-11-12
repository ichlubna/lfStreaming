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
    std::string input;
    std::unique_ptr<Renderer> renderer;
    void init();
    void initCodec();
    std::vector<glm::vec2> parseTrajectory(std::string textTrajectory);
    void storeImage(std::vector<uint8_t> *data, glm::uvec2 resolution, std::string path);
};
