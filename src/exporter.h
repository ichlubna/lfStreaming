#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Exporter
{
    public:
        void exportImage(CUdeviceptr data, size_t pitch, glm::ivec2 size, std::string path);
        void storeImage(std::vector<uint8_t> *data, glm::uvec2 resolution, std::string path, bool flipY = false);

    private:
        std::pair<cudaArray *, cudaSurfaceObject_t> createSurfaceObject(glm::ivec2 size);
};
