#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>

namespace PerPixelInterpolation
{
constexpr size_t INPUT_COUNT{4};
void perPixel(std::vector<CUdeviceptr> inFrames, std::vector<float> inWeights, std::vector<size_t> inPitches, uint8_t *result, int2 resolution, int pitch);
}

namespace Conversion
{
void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int pitch);
}
