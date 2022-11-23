#include <cuda_runtime.h>

class CudaGLInterop
{
    public:
    CudaGLInterop(){};
    ~CudaGLInterop();
    void setTexture(unsigned int inputTexture);
    

    private:
    bool registered{false};
    unsigned int texture;
    cudaGraphicsResource_t graphicsResource;
    void unset();
};
