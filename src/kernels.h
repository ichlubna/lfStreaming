#include <cuda_runtime.h>

namespace PerPixelInterpolation
{
void perPixel(const void * const* frames, uint8_t *result, int2 resolution, int pitch);
}

namespace Conversion
{
void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int pitch);
}
