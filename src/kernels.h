#include <cuda_runtime.h>

namespace Conversion
{
    void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int pitch);
}
