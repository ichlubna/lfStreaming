#include <sstream>
#include <stdexcept>
#include <fstream>
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
    renderer->init();
    while(renderer->ready())
        renderer->render();
}

std::vector<glm::vec2> Decoder::parseTrajectory(std::string textTrajectory)
{ 
    constexpr char pairDelimiter{','};
    constexpr char valueDelimiter{'_'};

    std::vector<glm::vec2> trajectory;
    std::istringstream stream(textTrajectory);
    std::string pair;
    std::string value;
    while (std::getline(stream, pair, pairDelimiter))
    {
        glm::vec2 position;
        int i{0};
        std::istringstream pairStream(pair);
        while (std::getline(pairStream, value, valueDelimiter))
        {
            position[i] = std::stof(value);
            if(position[i] < 0 || position[i] > 1)
                throw std::runtime_error("Trajectory values are not normalized!");
            i++;
        }
        trajectory.push_back(position);
    }
    return trajectory;
}

void Decoder::storeImage(std::vector<uint8_t> *data, glm::uvec2 resolution, std::string path)
{
  	std::ofstream fs(path, std::ios::out | std::ios::binary);
    if (!fs.is_open())
        throw std::runtime_error("Cannot open the file "+path+" for storing.");
    constexpr char const *BINARY_PPM{"P6"};
    constexpr size_t MAX_VAL{255};
    fs << BINARY_PPM << std::endl;
	fs << "#Exported with Light field streaming framework" << std::endl;
	fs << resolution.x << " " << resolution.y << std::endl;
	fs << MAX_VAL << std::endl;

    size_t pxId{0};
    for(size_t i=0; i<data->size(); i++)
    {
        if(pxId != 3)
            fs << data->at(i);
        pxId++;
        if(pxId > 3)
            pxId = 0;
    }
}

void Decoder::decodeAndStore(std::string trajectory)
{
    auto positions = parseTrajectory(trajectory);
}
