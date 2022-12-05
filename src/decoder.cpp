#include <sstream>
#include <stdexcept>
#include <string>
#include "decoder.h"
#include "exporter.h"

Decoder::Decoder(std::string inputPath, size_t startFrame) : renderer{std::make_unique<Renderer>()}, videoDecoder{std::make_unique<VideoDecoder>(inputPath)}, interop{std::make_unique<CudaGLInterop>()}
{
    videoDecoder->seek(startFrame);
}

void Decoder::decodeAndPlay()
{
    auto resolution = videoDecoder->getResolution();
    renderer->init();
    interop->setTexture(renderer->getTexture(resolution), resolution);
    videoDecoder->decodeFrame({0,0});
    videoDecoder->decodeFrame({0,0});
    videoDecoder->decodeFrame({0,0});
    videoDecoder->decodeFrame({0,0});
    auto frames = videoDecoder->getFrames();
    interop->copyData((*frames)[0].frame, (*frames)[0].pitch);
    while(renderer->ready())
    {
        renderer->render();
    }
}

std::vector<glm::vec2> Decoder::parseTrajectory(std::string textTrajectory) const
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

Decoder::SelectedFrames Decoder::pickFromGrid(glm::uvec2 gridSize, glm::vec2 position) const
{
    SelectedFrames frames;
    glm::vec2 gridPosition{glm::vec2(gridSize - 1u) *position};
    glm::ivec2 downCoords{glm::floor(gridPosition)};
    glm::ivec2 upCoords{glm::ceil(gridPosition)};

    float weight;
    glm::ivec2 currentCoords{0, 0};
    glm::vec2 unitPos{glm::fract(gridPosition)};

    currentCoords = {downCoords};
    weight = (1 - unitPos.x) * (1 - unitPos.y);
    frames.topLeft = {currentCoords, weight};

    currentCoords = {upCoords.x, downCoords.y};
    weight = unitPos.x * (1 - unitPos.y);
    frames.topLeft = {currentCoords, weight};

    currentCoords = {downCoords.x, upCoords.y};
    weight = (1 - unitPos.x) * unitPos.y;
    frames.topLeft = {currentCoords, weight};

    currentCoords = {upCoords};
    weight = unitPos.x * unitPos.y;
    frames.topLeft = {currentCoords, weight};    

    return frames;
}

void Decoder::interpolateView(glm::vec2 position)
{
}

void Decoder::decodeAndStore(std::string trajectory)
{
    auto positions = parseTrajectory(trajectory);
    Exporter exporter;
    for(auto const &position : positions)
    {
        std::cout << "Decoding view " << std::endl;
        videoDecoder->decodeFrame({0,0});
        videoDecoder->decodeFrame({0,0});
        videoDecoder->decodeFrame({0,0});
        videoDecoder->flush();
        while(!videoDecoder->allFramesReady())
        {}
        std::cout << "Interpolating view " << position.x << "_" << position.y << std::endl;
        auto frames = videoDecoder->getFrames();
        std::string fileName{std::to_string(position.x) + "_" + std::to_string(position.y) + ".ppm"};
        std::cout << "Storing result to "+fileName << std::endl;
        exporter.exportImage((*frames)[0].frame, (*frames)[0].pitch, videoDecoder->getResolution(), fileName);
    }
    
}
