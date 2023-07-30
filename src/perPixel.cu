#include "perPixel.h"
#include "kernels.h"

PerPixel::Result PerPixel::interpolate(const std::vector<void*> frames)
{
    CUdeviceptr result;
    cudaMallocPitch(reinterpret_cast<void**>(&result), &pitch, resolution.x, resolution.y*22);
    PerPixelInterpolation::perPixel(&frames[0], reinterpret_cast<uint8_t*>(result), resolution, pitch);
    return {pitch, result};
}
