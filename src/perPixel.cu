#include "perPixel.h"
#include "kernels.h"
#include <stdexcept>

PerPixel::PerPixel(glm::ivec2 res) : resolution{res.x, res.y}, pixelCount{static_cast<size_t>(res.x*res.y)}
{
    if(cudaMallocPitch(reinterpret_cast<void**>(&result), &pitch, resolution.x, resolution.y+(resolution.y+1)/2) != cudaSuccess)
        throw std::runtime_error("Cannot allocate result frame for per pixel interpolation.");
};

#include <iostream>
PerPixel::Result PerPixel::interpolate(PerPixel::InputFrames input)
{
    PerPixelInterpolation::perPixel(input.frames, input.weights, input.pitches, reinterpret_cast<uint8_t*>(result), resolution, pitch);
    return {pitch, reinterpret_cast<CUdeviceptr>(result)};
}
