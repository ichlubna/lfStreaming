#include "decoder.h"

Decoder::Decoder(std::string inputPath) : input{inputPath}, renderer{std::make_unique<Renderer>()}
{
    init();
}

void Decoder::init()
{
    initCodec();
}

void Decoder::initCodec()
{

}

void Decoder::decodeAndPlay()
{
    while(!renderer->endSignaled())
        renderer->render();
}
