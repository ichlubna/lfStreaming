#include <stdexcept>
#include "cudaGLInterop.h"
#include "kernels.cu"

CudaGLInterop::~CudaGLInterop()
{
    unset();
}

void CudaGLInterop::unset()
{
    if(registered)
        if(cudaGraphicsUnregisterResource(graphicsResource) != cudaSuccess)
            throw std::runtime_error("Cannot unregister graphics resource.");
    registered = false;
}

void CudaGLInterop::setTexture(GLuint inputTexture, glm::ivec2 textureResolution)
{
    unset();
    textureGL = inputTexture;
    resolution = textureResolution;
    if(cudaGraphicsGLRegisterImage(&graphicsResource, textureGL, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess)
        throw std::runtime_error("Cannot register graphics resource.");
}

void CudaGLInterop::copyData(CUdeviceptr input)
{
    cudaGraphicsMapResources(1, &graphicsResource);
    cudaArray_t output;
    cudaGraphicsSubResourceGetMappedArray(&output, graphicsResource, 0, 0);
    cudaResourceDesc resourceDesc;
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = output;
    cudaSurfaceObject_t surface;
    cudaCreateSurfaceObject(&surface, &resourceDesc); 
    Conversion::NV12ToRGBA(reinterpret_cast<uint8_t*>(input), surface, {resolution.x, resolution.y});
    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &graphicsResource);
    cudaStreamSynchronize(0); 
}
