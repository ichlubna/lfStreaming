#include <stdexcept>
#include <cuda_gl_interop.h>
#include "cudaGLInterop.h"

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

void CudaGLInterop::setTexture(unsigned int inputTexture)
{
    unset();
    texture = inputTexture;
    if(cudaGraphicsGLRegisterImage(&graphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess)
        throw std::runtime_error("Cannot register graphics resource.");
}
