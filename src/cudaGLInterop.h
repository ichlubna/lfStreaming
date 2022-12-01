#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

class CudaGLInterop
{
    public:
    CudaGLInterop(){};
    ~CudaGLInterop();
    void setTexture(GLuint inputTexture, glm::ivec2 textureResolution);
    void copyData(CUdeviceptr input, int pitch); 

    private:
    bool registered{false};
    GLuint textureGL;
    glm::ivec2 resolution;
    cudaGraphicsResource_t graphicsResource;
    void unset();
};
