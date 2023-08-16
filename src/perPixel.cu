#include "perPixel.h"
#include "kernels.h"
#include <stdexcept>

PerPixel::PerPixel(glm::ivec2 res) : resolution{res.x, res.y}, pixelCount{static_cast<size_t>(res.x*res.y)}
{
    if(cudaMallocPitch(reinterpret_cast<void**>(&result), &pitch, resolution.x, resolution.y+(resolution.y+1)/2) != cudaSuccess)
    //pitch = 2048;
    //if(cudaMalloc(reinterpret_cast<void**>(&result), pitch*(resolution.y+(resolution.y+1)/2)) != cudaSuccess)
        throw std::runtime_error("Cannot allocate result frame for per pixel interpolation.");
};

PerPixel::Result PerPixel::interpolate(PerPixel::InputFrames input)
{
    std::vector<float2> inOffsets;
    float aspect = (static_cast<float>(resolution.x)/resolution.y)/input.aspect;
    for(const auto &o : input.offsets)
        inOffsets.push_back({(o.x/8.0f)*resolution.x, ((o.y*aspect)/8.0f)*resolution.x}); 
    PerPixelInterpolation::perPixel(    input.frames, input.weights,
                                        inOffsets, input.pitches,
                                         reinterpret_cast<uint8_t*>(result),
                                         input.inverseWeightSum,
                                         resolution, pitch, 
                                         {input.focusRange.x, input.focusRange.y});

    return {pitch, reinterpret_cast<CUdeviceptr>(result)};
}
