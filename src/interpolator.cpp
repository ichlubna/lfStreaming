#include <stdexcept>
#include <vector>
#include <dlfcn.h>
#include "interpolator.h"

Interpolator::Interpolator(glm::ivec2 inputResolution) : resolution{inputResolution}
{
    init();
}

Interpolator::~Interpolator()
{
    NvOFFRUCDestroy(fruc);

    NvOFFRUC_UNREGISTER_RESOURCE_PARAM unregisterParam{};
    std::vector<void*> resources; 
    for(size_t i=0; i<resources.size(); i++)
        unregisterParam.pArrResource[i] = resources[i];
    unregisterParam.uiCount = resources.size();
    NvOFFRUCUnregisterResource(fruc, &unregisterParam);
}

void* Interpolator::getProc(void *handle, const char *symbol)
{
    if constexpr (OS == LINUX)
        return dlsym(handle, symbol);
    else
        throw std::runtime_error("Used OS support not implemented.");
}

void Interpolator::loadLibrary()
{
    void *library = dlopen(NvOFFRCULibraryPath, RTLD_LAZY);
    if(!library)
        throw std::runtime_error("Fruc library can't be dynamically loaded.");

    NvOFFRUCCreate = (PtrToFuncNvOFFRUCCreate)getProc(library, CreateProcName);
    NvOFFRUCRegisterResource = (PtrToFuncNvOFFRUCRegisterResource)getProc(library, RegisterResourceProcName);
    NvOFFRUCUnregisterResource = (PtrToFuncNvOFFRUCUnregisterResource)getProc(library, UnregisterResourceProcName);
    NvOFFRUCProcess = (PtrToFuncNvOFFRUCProcess)getProc(library, ProcessProcName);
    NvOFFRUCDestroy = (PtrToFuncNvOFFRUCDestroy)getProc(library, DestroyProcName);
}

void Interpolator::registerResources()
{
    //preallocate memory and copy inputs there?
    std::vector<void*> resources; 
    NvOFFRUC_REGISTER_RESOURCE_PARAM registerParam{};
    for(size_t i=0; i<resources.size(); i++)
        registerParam.pArrResource[i] = resources[i];
    registerParam.uiCount = resources.size();
                        
    if(NvOFFRUCRegisterResource(fruc, &registerParam) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot register resources.");
    //use the same struct to unregister?
}

void Interpolator::process(float factor, bool updateOnly)
{
    void *inFrame{nullptr};
    NvOFFRUC_PROCESS_IN_PARAMS inParams{};
    inParams.stFrameDataInput.pFrame = inFrame;
    inParams.stFrameDataInput.nTimeStamp = frameNumber;
    inParams.bSkipWarp = updateOnly;

    void *outFrame{nullptr};
    NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
    outParams.stFrameDataOutput.pFrame = outFrame;
    outParams.stFrameDataOutput.nTimeStamp = frameNumber-1+factor;

    frameNumber++;

    if(NvOFFRUCProcess(fruc, &inParams, &outParams) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot interpolate frames.");

}

void Interpolator::initInterpolation()
{
    NvOFFRUC_CREATE_PARAM createParams{};
    createParams.pDevice = nullptr;
    createParams.uiWidth = resolution.x;
    createParams.uiHeight = resolution.y;;
    createParams.eResourceType = NvOFFRUCResourceType::CudaResource;
    createParams.eSurfaceFormat = NvOFFRUCSurfaceFormat::NV12Surface;
    createParams.eCUDAResourceType  = NvOFFRUCCUDAResourceType::CudaResourceCuDevicePtr;
    
    if(NvOFFRUCCreate(&createParams, &fruc) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot initialize FRUC.");
}

void Interpolator::init()
{
    loadLibrary();
    initInterpolation(); 
}


