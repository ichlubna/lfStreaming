#include <string>
#include <memory>
#include "renderer.h"

class Decoder
{
    public:
    Decoder(std::string input);
    void decodeAndPlay();

    private:
    std::string input;
    std::unique_ptr<Renderer> renderer;
    void init();
    void initCodec();
};
