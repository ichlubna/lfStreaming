#include <glm/detail/qualifier.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include "decoder.h"
#include "exporter.h"
#include "kernels.h"

Decoder::Decoder(std::string inputPath, size_t startFrame) : renderer{std::make_unique<Renderer>()}, videoDecoder{std::make_unique<VideoDecoder>(inputPath)}, interop{std::make_unique<CudaGLInterop>()}
{
    interpolator = std::make_unique<Interpolator>(videoDecoder->getResolution());
    perPixel = std::make_unique<PerPixel>(videoDecoder->getResolution());
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
    else if(method == "PP")
    {
        interpolationOrder = SelectedFrames::TOP_BOTTOM;
        usePerPixel = true;
    }
    else
    {
        interpolationOrder = SelectedFrames::TOP_BOTTOM;
        std::cout << "Selected interpolation method unknown - using default OF_TB." << std::endl;
    }
}

glm::vec2 Decoder::cameraPosition()
{
    auto mouse = renderer->getMousePosition();
    mouse = {mouse.y, mouse.x};
    constexpr float DELTA{0.0000001};
    return glm::clamp(mouse, {0 + DELTA, 0 + DELTA}, {1 - DELTA, 1 - DELTA});
}

void Decoder::decodeAndPlay(float framerate)
{
    auto resolution = videoDecoder->getResolution();
    renderer->init();
    interop->setTexture(renderer->getTexture(resolution), resolution);
    Timer<true,false> playbackTimer;
    const int frameTime = 1000.0f/framerate;
    const size_t length = videoDecoder->getTimeLength();
    while(renderer->ready())
    {
        if(framerate > 0)
            playbackTimer.start();
       
        if(length > 1 || renderer->mouseChanged())
        {
            auto result = decodeAndInterpolate(cameraPosition());
            if(!renderer->isPaused())
                videoDecoder->incrementTime();
            interop->copyData(result.frame, result.pitch);
        }
        renderer->render();

        if(framerate > 0)
        {
            auto elapsed = playbackTimer.stop();
            int delay = glm::round(frameTime - elapsed.cpuTime);
            if(delay > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
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
    glm::vec2 maxCoords{gridSize - 1u};
    glm::vec2 gridPosition{maxCoords*position};
    glm::ivec2 downCoords{glm::floor(gridPosition)};
    glm::ivec2 upCoords{glm::ceil(gridPosition)};

    float weight;
    glm::ivec2 currentCoords{0, 0};
    glm::vec2 unitPos{glm::fract(gridPosition)};

    currentCoords = {downCoords};
    weight = (1 - unitPos.x) * (1 - unitPos.y);
    frames[TOP_LEFT] = {currentCoords, weight, gridPosition-glm::vec2(currentCoords)};

    currentCoords = {upCoords.x, downCoords.y};
    weight = unitPos.x * (1 - unitPos.y);
    frames[TOP_RIGHT] = {currentCoords, weight, gridPosition-glm::vec2(currentCoords)};

    currentCoords = {downCoords.x, upCoords.y};
    weight = (1 - unitPos.x) * unitPos.y;
    frames[BOTTOM_LEFT] = {currentCoords, weight, gridPosition-glm::vec2(currentCoords)};

    currentCoords = {upCoords};
    weight = unitPos.x * unitPos.y;
    frames[BOTTOM_RIGHT] = {currentCoords, weight, gridPosition-glm::vec2(currentCoords)};
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
    Timer<true,true> timer;
    if constexpr (measure)
    {
        std::cout << "Decoding the frames..." << std::endl;
        timer.start();
    }

    videoDecoder->clearBuffer();
    framePicker.compute(videoDecoder->getColsRows(), position);
    auto guide = framePicker.guide(interpolationOrder);
    for(size_t i = 0; i < guide.FRAME_COUNT; i++)
        videoDecoder->decodeFrame(guide.positions[i]);
    while(!videoDecoder->allFramesReady()) {}

    auto framePtrs = videoDecoder->getFramePointers();
    auto frames = videoDecoder->getFrames();

    if constexpr (measure)
    {
        timer.stop().printElapsed();
    }
    if(usePerPixel)
        return interpolatePerPixel<measure>(frames, framePtrs, guide);    
    return interpolateOptical<measure>(frames, framePtrs, guide);    
}

template<bool measure>
Decoder::InterpolationResult Decoder::interpolatePerPixel(const std::vector<VideoDecoder::DecodedFrame> *frames, const std::vector<void*> framePtrs, Decoder::SelectedFrames::InterpolationInfo guide)
{
    Timer<true,true> timer;
    if constexpr (measure)
    {
        std::cout << "Interpolating new view..." << std::endl;
        timer.start();
    }
    
    PerPixel::InputFrames input;
    for(size_t i=0; i<PerPixel::InputFrames::COUNT; i++)
    {
        input.frames.push_back(frames->at(i).frame);
        input.weights.push_back(framePicker.frames[i].weight);
        input.inverseWeightSum += input.weights.back();
        input.pitches.push_back(frames->at(i).pitch);
        input.offsets.push_back(framePicker.frames[i].offset);
    } 
    input.inverseWeightSum = 1.0f/input.inverseWeightSum;
    input.aspect = videoDecoder->getGridAspect();
    input.focusRange = videoDecoder->getFocusRange(); 
    auto result = perPixel->interpolate(input);    

    if constexpr (measure)
    {
        timer.stop().printElapsed();
    }
    return {result.result, result.pitch};
}


template<bool measure>
Decoder::InterpolationResult Decoder::interpolateOptical(const std::vector<VideoDecoder::DecodedFrame> *frames, const std::vector<void*> framePtrs, Decoder::SelectedFrames::InterpolationInfo guide)
{
    Timer<true,true> timer;
    if constexpr (measure)
    {
        std::cout << "Interpolating new view..." << std::endl;
        timer.start();
    }
    interpolator->registerResources(&framePtrs);

    for(size_t i = 0; i < guide.WEIGHTS_COUNT; i++)
    {
        size_t frameID = i * 2;
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

    if constexpr (measure)
    {
        timer.stop().printElapsed();
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
        auto result = decodeAndInterpolate<true>(position);
        std::string fileName{std::filesystem::path(outputPath) / (std::to_string(position.x) + "_" + std::to_string(position.y) + ".ppm")};
        std::cout << "Storing result to " + fileName << std::endl;
        exporter.exportImage(result.frame, result.pitch, videoDecoder->getResolution(), fileName);
    }
}
