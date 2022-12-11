#include <glm/detail/qualifier.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include "decoder.h"
#include "exporter.h"
#include "kernels.h"
#include "timer.h"

Decoder::Decoder(std::string inputPath, size_t startFrame) : renderer{std::make_unique<Renderer>()}, videoDecoder{std::make_unique<VideoDecoder>(inputPath)}, interop{std::make_unique<CudaGLInterop>()}
{
    interpolator = std::make_unique<Interpolator>(videoDecoder->getResolution());
    videoDecoder->seek(startFrame);
    prepareFrames();
    intermediatePtrs = getIntermediatePtrs();
    interpolator->registerResources(&intermediatePtrs);
}

Decoder::~Decoder()
{
    interpolator->unregisterResources(&intermediatePtrs);
}

void Decoder::setInterpolationMethod(std::string method)
{
    if(method == "OF_TB")
        interpolationOrder = SelectedFrames::TOP_BOTTOM;
    else if(method == "OF_LR")
        interpolationOrder = SelectedFrames::LEFT_RIGHT;
    else if(method == "OF_D")
        interpolationOrder = SelectedFrames::DIAGONAL;
    else
    {
        interpolationOrder = SelectedFrames::TOP_BOTTOM;
        std::cout << "Selected interpolation method unknown - using default OF_TB." << std::endl;
    }
}

glm::vec2 Decoder::cameraPosition()
{
    auto mouse = renderer->getMousePosition();
    constexpr float DELTA{0.0000001};
    return glm::clamp(mouse, {0 + DELTA, 0 + DELTA}, {1 - DELTA, 1 - DELTA});
}

void Decoder::decodeAndPlay()
{
    auto resolution = videoDecoder->getResolution();
    renderer->init();
    interop->setTexture(renderer->getTexture(resolution), resolution);
    while(renderer->ready())
    {
        if(renderer->mouseChanged())
        {
            auto result = decodeAndInterpolate(cameraPosition());
            interop->copyData(result.frame, result.pitch);
        }
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
    while(std::getline(stream, pair, pairDelimiter))
    {
        glm::vec2 position;
        int i{0};
        std::istringstream pairStream(pair);
        while(std::getline(pairStream, value, valueDelimiter))
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

void Decoder::SelectedFrames::compute(glm::uvec2 gridSize, glm::vec2 position)
{
    glm::vec2 gridPosition{glm::vec2(gridSize - 1u) *position};
    glm::ivec2 downCoords{glm::floor(gridPosition)};
    glm::ivec2 upCoords{glm::ceil(gridPosition)};

    float weight;
    glm::ivec2 currentCoords{0, 0};
    glm::vec2 unitPos{glm::fract(gridPosition)};

    currentCoords = {downCoords};
    weight = (1 - unitPos.x) * (1 - unitPos.y);
    frames[TOP_LEFT] = {currentCoords, weight};

    currentCoords = {upCoords.x, downCoords.y};
    weight = unitPos.x * (1 - unitPos.y);
    frames[TOP_RIGHT] = {currentCoords, weight};

    currentCoords = {downCoords.x, upCoords.y};
    weight = (1 - unitPos.x) * unitPos.y;
    frames[BOTTOM_LEFT] = {currentCoords, weight};

    currentCoords = {upCoords};
    weight = unitPos.x * unitPos.y;
    frames[BOTTOM_RIGHT] = {currentCoords, weight};
}

Decoder::SelectedFrames::InterpolationInfo Decoder::SelectedFrames::guide(Order order)
{
    InterpolationInfo info;
    auto &selectedOrder = orderIDs[order];
    for(size_t i = 0; i < InterpolationInfo::FRAME_COUNT; i++)
        info.positions[i] = frames[selectedOrder[i]].coord;
    info.weights[0] = interWeight(selectedOrder[0], selectedOrder[1]);
    info.weights[1] = interWeight(selectedOrder[2], selectedOrder[3]);
    info.weights[2] = interWeight(order);
    return info;
}

Decoder::IntermediateFrame::IntermediateFrame(glm::ivec2 resolution)
{
    if(cuMemAllocPitch(&frame, &pitch, resolution.x, resolution.y * 2, 4) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot allocate memory for interpolation results.");
}

void Decoder::prepareFrames()
{
    constexpr size_t INTERMEDIATE_COUNT{3};
    for(size_t i = 0; i < INTERMEDIATE_COUNT; i++)
        intermediateFrames.push_back(IntermediateFrame(videoDecoder->getResolution()));
}

std::vector<void *> Decoder::getIntermediatePtrs()
{
    std::vector<void *> ptrs;
    for(auto &frame : intermediateFrames)
        ptrs.push_back(frame.ptr());
    return ptrs;
}

template<bool measure>
Decoder::InterpolationResult Decoder::decodeAndInterpolate(glm::vec2 position)
{
    if constexpr (measure)
        Timer<true,true> timer;

    videoDecoder->clearBuffer();
    framePicker.compute(videoDecoder->getColsRows(), position);
    auto guide = framePicker.guide(interpolationOrder);
    for(size_t i = 0; i < guide.FRAME_COUNT; i++)
        videoDecoder->decodeFrame(guide.positions[i]);
    videoDecoder->flush();
    while(!videoDecoder->allFramesReady()) {}

    auto framePtrs = videoDecoder->getFramePointers();
    auto frames = videoDecoder->getFrames();
    interpolator->registerResources(&framePtrs);

    for(size_t i = 0; i < guide.WEIGHTS_COUNT; i++)
    {
        size_t frameID = 1 + i * 2;
        Interpolator::Pair pair;
        if(i != guide.WEIGHTS_COUNT - 1)
        {
            pair.first = framePtrs[frameID];
            pair.second = framePtrs[frameID + 1];
            pair.pitch = {frames->at(frameID).pitch, frames->at(frameID + 1).pitch};
        }
        else
        {
            pair.first = intermediatePtrs[0];
            pair.second = intermediatePtrs[1];
            pair.pitch = {intermediateFrames[0].pitch, intermediateFrames[1].pitch};
        }
        pair.output = intermediatePtrs[i];
        pair.weight = guide.weights[i];
        interpolator->interpolate({pair});
    }

    interpolator->unregisterResources(&framePtrs);
    auto result = intermediateFrames[guide.WEIGHTS_COUNT - 1];
    return {result.frame, result.pitch};
}

void Decoder::decodeAndStore(std::string trajectory, std::string outputPath)
{
    auto positions = parseTrajectory(trajectory);
    Exporter exporter;

    for(auto const &position : positions)
    {
        std::cout << "____________________________" << std::endl;
        std::cout << "Processing view " << position.x << "_" << position.y << std::endl;
        auto result = decodeAndInterpolate(position);
        std::string fileName{std::filesystem::path(outputPath) / (std::to_string(position.x) + "_" + std::to_string(position.y) + ".ppm")};
        std::cout << "Storing result to " + fileName << std::endl;
        exporter.exportImage(result.frame, result.pitch, videoDecoder->getResolution(), fileName);
    }
}
