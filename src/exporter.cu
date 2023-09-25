#include <fstream>
#include "exporter.h"
#include "kernels.h"

void Exporter::storeImage(std::vector<uint8_t> *data, glm::uvec2 resolution, std::string path, bool flipY) const
{
    std::ofstream fs(path, std::ios::out | std::ios::binary);
    if(!fs.is_open())
        throw std::runtime_error("Cannot open the file " + path + " for storing.");
    constexpr char const *BINARY_PPM{"P6"};
    constexpr size_t MAX_VAL{255};
    fs << BINARY_PPM << std::endl;
    fs << "#Exported with Light field streaming framework" << std::endl;
    fs << resolution.x << " " << resolution.y << std::endl;
    fs << MAX_VAL << std::endl;

    constexpr size_t CHANNELS{4};
    for(size_t y = 0; y < resolution.y; y++)
        for(size_t x = 0; x < resolution.x; x++)
        {
            size_t yy = y;
            if(flipY)
                yy = resolution.y - 1 - y;
            size_t id = yy * resolution.x * CHANNELS + x * CHANNELS;
            for(size_t c = 0; c < CHANNELS - 1; c++)
                fs << data->at(id + c);
        }
}

std::pair<cudaArray *, cudaSurfaceObject_t> Exporter::createSurfaceObject(glm::ivec2 size) const
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y, cudaArraySurfaceLoadStore);
    cudaResourceDesc surfRes{};
    //memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = arr;
    cudaSurfaceObject_t surfObj{0};
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {arr, surfObj};
}

void Exporter::exportImage(CUdeviceptr data, size_t pitch, glm::ivec2 size, std::string path) const
{
    auto surfaceData = createSurfaceObject(size);
    Conversion::NV12ToRGBA(reinterpret_cast<uint8_t *>(data), surfaceData.second, {size.x, size.y}, pitch);
    constexpr size_t CHANNELS{4};
    std::vector<uint8_t> imgData(size.x * size.y * CHANNELS);
    cudaMemcpy2DFromArray(imgData.data(), size.x * CHANNELS, surfaceData.first, 0, 0, size.x * CHANNELS, size.y, cudaMemcpyDeviceToHost);
    storeImage(&imgData, size, path, true);
}
