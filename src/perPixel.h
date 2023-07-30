#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class PerPixel
{
    public:
    class Result
    {
        public:
        size_t pitch;
        CUdeviceptr result;
    };
    PerPixel(glm::ivec2 res) : resolution{res.x, res.y}, pixelCount{static_cast<size_t>(res.x*res.y)}{};
    static constexpr size_t FRAMES_COUNT{4};
    Result interpolate(const std::vector<void*> frames);

    private:
    uint8_t *result;
    int2 resolution;
    size_t pixelCount;
    size_t pitch;
};
