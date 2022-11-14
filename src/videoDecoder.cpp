#include <cstring>
#include <exception>
#include "videoDecoder.h"

VideoDecoder::VideoDecoder(std::string file) : demuxer{std::make_unique<Muxing::Demuxer>(file)}
{
    init();
}

cudaVideoCodec VideoDecoder::getCodec()
{
    cudaVideoCodec codec = cudaVideoCodec_HEVC;
    //if(demuxer->data.format() == Muxing::EncodedData::Format::AV1)
    //    codec = cudaVideoCodec_AV1;
    return codec;
}
#include <iostream>

void initContext()
{
    cuInit(0);
    //int deviceCount{0};
    //cuDeviceGetCount(&deviceCount);
    CUdevice device = 0;
    CUcontext cuContext = NULL;
    cuCtxCreate(&cuContext, 0, device);
}

void VideoDecoder::checkGPU()
{
    CUVIDDECODECAPS capabilities{};
    capabilities.eCodecType = getCodec();;
    capabilities.eChromaFormat = chromaFormat;
    capabilities.nBitDepthMinus8 = 0;

    if(cuvidGetDecoderCaps(&capabilities) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot check decoder capabilities.");
 
    if(!capabilities.bIsSupported)
        throw std::runtime_error("Codec not available.");
    
    if ((demuxer->data.resolution().x > capabilities.nMaxWidth) || (demuxer->data.resolution().y > capabilities.nMaxHeight))
        throw std::runtime_error("Video resolution not supported.");
}

void VideoDecoder::createDecoder()
{;
    constexpr int DECODED_COUNT{5};
    CUVIDDECODECREATEINFO videoDecodeCreateInfo{};
    videoDecodeCreateInfo.CodecType = getCodec();
    videoDecodeCreateInfo.ChromaFormat = chromaFormat;
    videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    videoDecodeCreateInfo.bitDepthMinus8 = 0;
    videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    videoDecodeCreateInfo.ulNumOutputSurfaces = DECODED_COUNT;
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = DECODED_COUNT;
    videoDecodeCreateInfo.ulWidth = demuxer->data.resolution().x;
    videoDecodeCreateInfo.ulHeight = demuxer->data.resolution().y;
    videoDecodeCreateInfo.ulTargetWidth = demuxer->data.resolution().x;
    videoDecodeCreateInfo.ulTargetHeight = demuxer->data.resolution().y;
    if(cuvidCreateDecoder(&decoder, &videoDecodeCreateInfo) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot create the decoder.");

}

void VideoDecoder::init()
{
    initContext();
    checkGPU();
    createDecoder();
}
