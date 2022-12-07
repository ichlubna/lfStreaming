#include <cstring>
#include <exception>
#include <memory>
#include "videoDecoder.h"

#include <stdexcept>

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

void VideoDecoder::createParser()
{
    CUVIDPARSERPARAMS videoParserParameters{};
    videoParserParameters.CodecType = getCodec();
    videoParserParameters.ulMaxNumDecodeSurfaces = DECODED_COUNT;
    videoParserParameters.ulMaxDisplayDelay = 0;
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = videoSequenceCallback;
    videoParserParameters.pfnDecodePicture = decodePictureCallback;
    videoParserParameters.pfnDisplayPicture = displayPictureCallback;
    if(cuvidCreateVideoParser(&parser, &videoParserParameters) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot create the parser.");
}

int VideoDecoder::videoSequence(CUVIDEOFORMAT *format)
{
    return DECODER_CALLBACK_SUCCESS;
} 

int VideoDecoder::decodePicture(CUVIDPICPARAMS *picParams)
{  
    if(cuvidDecodePicture(decoder, picParams) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot decode picture.");
    return DECODER_CALLBACK_SUCCESS;
}

void VideoDecoder::prepareFrame(int timestamp, int pictureID, CUVIDPROCPARAMS params)
{
    int finalID = timestamp % FRAME_COUNT;
    frames[finalID] = DecodedFrame(decoder);
    cuvidMapVideoFrame(decoder, pictureID, &(frames[finalID].frame),  &(frames[finalID].pitch), &params);
}
 
int VideoDecoder::displayPicture(CUVIDPARSERDISPINFO *dispInfo)
{
    CUVIDPROCPARAMS videoProcessingParameters{};
    videoProcessingParameters.progressive_frame = dispInfo->progressive_frame;
    videoProcessingParameters.second_field = dispInfo->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = dispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = dispInfo->repeat_first_field < 0;
    videoProcessingParameters.output_stream = 0;
    CUVIDGETDECODESTATUS status{};
    if(cuvidGetDecodeStatus(decoder, dispInfo->picture_index, &status) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot check decoding status.");
    if (status.decodeStatus == cuvidDecodeStatus_Error || status.decodeStatus == cuvidDecodeStatus_Error_Concealed)
        throw std::runtime_error("Cannot get decoded frame.");
  
     
    prepareFrame(dispInfo->timestamp, dispInfo->picture_index, videoProcessingParameters); 
    return DECODER_CALLBACK_SUCCESS;
} 

bool VideoDecoder::allFramesReady()
{
    bool ready{true};
    for(auto const &frame : frames)
        ready = ready && (frame.frame != 0);
    return ready;
}

std::vector<void*> VideoDecoder::getFramePointers()
{
    std::vector<void*> ptrs(frames.size());
    for(size_t i=0; i<frames.size(); i++)
        ptrs[i] = reinterpret_cast<void*>(&(frames[i].frame)); 
    return ptrs; 
}

void VideoDecoder::incrementTime()
{
    auto newTime = time+1;
    if(newTime >= demuxer->data.timeFrameCount())
         newTime=0;
    seek(newTime);
}

void VideoDecoder::seek(size_t newTime)
{
    time = newTime;
    decode(demuxer->getReferencePacket(time));
}

void VideoDecoder::decodeFrame(glm::ivec2 position)
{
    auto packet = demuxer->getPacket({position, time});
    decode(packet);
}

void VideoDecoder::flush()
{
    decode({nullptr,0});
}

void VideoDecoder::decode(Muxing::Demuxer::PacketPointer packetPointer)
{
    CUVIDSOURCEDATAPACKET packet{};
    packet.payload = packetPointer.data;
    packet.payload_size = packetPointer.size;
    packet.flags = CUVID_PKT_TIMESTAMP;
    packet.timestamp = decodedNumber;
    decodedNumber++;
    if (packetPointer.size == 0) 
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    if(cuvidParseVideoData(parser, &packet) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot parse packet.");
}

void VideoDecoder::init()
{
    initContext();
    checkGPU();
    createDecoder();
    createParser();
}

VideoDecoder::~VideoDecoder()
{
    frames.clear();
    cuvidDestroyVideoParser(parser);
    cuvidDestroyDecoder(decoder);
}
