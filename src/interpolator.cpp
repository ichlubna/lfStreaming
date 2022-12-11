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
}

void *Interpolator::getProc(void *handle, const char *symbol)
{
    if constexpr(OS == LINUX)
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

void Interpolator::registerResources(const std::vector<void *> *resources)
{
    NvOFFRUC_REGISTER_RESOURCE_PARAM registerParam{};
    for(size_t i = 0; i < resources->size(); i++)
        registerParam.pArrResource[i] = resources->at(i);
    registerParam.uiCount = resources->size();

    if(NvOFFRUCRegisterResource(fruc, &registerParam) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot register resources.");
}

void Interpolator::unregisterResources(const std::vector<void *> *resources)
{
    NvOFFRUC_UNREGISTER_RESOURCE_PARAM unregisterParam{};
    for(size_t i = 0; i < resources->size(); i++)
        unregisterParam.pArrResource[i] = resources->at(i);
    unregisterParam.uiCount = resources->size();

    if(NvOFFRUCUnregisterResource(fruc, &unregisterParam) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot unregister resources.");
}

void Interpolator::process(float factor, bool updateOnly, void *input, void *output, size_t pitch)
{
    frameNumber++;
    NvOFFRUC_PROCESS_IN_PARAMS inParams{};
    inParams.stFrameDataInput.pFrame = input;
    inParams.stFrameDataInput.nTimeStamp = frameNumber;
    inParams.stFrameDataInput.nCuSurfacePitch = pitch;
    inParams.bSkipWarp = updateOnly;

    NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
    outParams.stFrameDataOutput.pFrame = output;
    outParams.stFrameDataOutput.nCuSurfacePitch = pitch;
    outParams.stFrameDataOutput.nTimeStamp = frameNumber - factor;
    bool dummy;
    outParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &dummy;

    if(NvOFFRUCProcess(fruc, &inParams, &outParams) != NvOFFRUC_SUCCESS)
        throw std::runtime_error("Cannot interpolate frames.");
}

void Interpolator::interpolate(std::vector<Pair> data)
{
    for(auto const &pair : data)
    {
        process(0.0, true, pair.first, pair.output, pair.pitch.x);
        process(pair.weight, false, pair.second, pair.output, pair.pitch.y);
    }
}

void Interpolator::initInterpolation()
{
    NvOFFRUC_CREATE_PARAM createParams{};
    createParams.pDevice = nullptr;
    createParams.uiWidth = 1024;//resolution.x;
    createParams.uiHeight = resolution.y;
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


