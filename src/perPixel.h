#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class PerPixel
{
    public:
    class InputFrames
    {
        public:
        static constexpr size_t COUNT{4};
        std::vector<CUdeviceptr> frames;
        std::vector<float> weights;
        std::vector<size_t> pitches;
        std::vector<glm::vec2> offsets;
    };

    class Result
    {
        public:
        size_t pitch;
        CUdeviceptr result;
    };
    PerPixel(glm::ivec2 res);
    Result interpolate(InputFrames input);

    private:
    uint8_t *result;
    int2 resolution;
    size_t pixelCount;
    size_t pitch;
};