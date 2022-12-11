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

    if((demuxer->data.resolution().x > capabilities.nMaxWidth) || (demuxer->data.resolution().y > capabilities.nMaxHeight))
        throw std::runtime_error("Video resolution not supported.");
}

void VideoDecoder::createDecoder()
{
    ;
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

int VideoDecoder::videoSequence([[maybe_unused]] CUVIDEOFORMAT *format)
{
    return DECODER_CALLBACK_SUCCESS;
}

int VideoDecoder::decodePicture(CUVIDPICPARAMS *picParams)
{
    if(cuvidDecodePicture(decoder, picParams) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot decode picture.");
    CUVIDPROCPARAMS videoProcessingParameters{};
    videoProcessingParameters.progressive_frame = 1;
    videoProcessingParameters.output_stream = 0;
    prepareFrame(picParams->CurrPicIdx, videoProcessingParameters);

    return DECODER_CALLBACK_SUCCESS;
}

void VideoDecoder::prepareFrame(int pictureID, CUVIDPROCPARAMS params)
{
    CUVIDGETDECODESTATUS status{};
    if(cuvidGetDecodeStatus(decoder, pictureID, &status) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot check decoding status.");
    if(status.decodeStatus == cuvidDecodeStatus_Error || status.decodeStatus == cuvidDecodeStatus_Error_Concealed)
        throw std::runtime_error("Cannot get decoded frame.");

    frames.emplace_back();
    frames.back().decoder = decoder;
    if(cuvidMapVideoFrame(decoder, pictureID, &(frames.back().frame),  &(frames.back().pitch), &params) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot map frame.");
}

int VideoDecoder::displayPicture([[maybe_unused]] CUVIDPARSERDISPINFO *dispInfo)
{
    //The timestamps were not correct here so I fetch the frames in decode callback

    /*CUVIDPROCPARAMS videoProcessingParameters{};
    videoProcessingParameters.progressive_frame = dispInfo->progressive_frame;
    videoProcessingParameters.output_stream = 0;
    */
//    prepareFrame(dispInfo->timestamp, dispInfo->picture_index, videoProcessingParameters);
    return DECODER_CALLBACK_SUCCESS;
}

bool VideoDecoder::allFramesReady() const
{
    return frames.size() > FRAME_COUNT;
    /*    bool ready{true};
        for(auto const &frame : frames)
            ready = ready && (frame.frame != 0);
        return ready;*/
}

void VideoDecoder::clearBuffer()
{
    frames.clear();
    frames.reserve(FRAME_COUNT + 5);
    initFrame();
}

const std::vector<void*> VideoDecoder::getFramePointers() const
{
    std::vector<void *> ptrs(frames.size());
    for(size_t i = 0; i < frames.size(); i++)
        ptrs[i] = const_cast<void*>(reinterpret_cast<const void*>(&(frames[i].frame)));
    return ptrs;
}

void VideoDecoder::incrementTime()
{
    auto newTime = time + 1;
    if(newTime >= demuxer->data.timeFrameCount())
        newTime = 0;
    seek(newTime);
}

void VideoDecoder::seek(size_t newTime)
{
    time = newTime;
    decodedNumber = 0;
}

void VideoDecoder::initFrame()
{
    decode(demuxer->getReferencePacket(time));
}

void VideoDecoder::decodeFrame(glm::ivec2 position)
{
    auto packet = demuxer->getPacket({position, time});
    decode(packet);
}

void VideoDecoder::flush()
{
    decode({nullptr, 0});
}

void VideoDecoder::decode(Muxing::Demuxer::PacketPointer packetPointer)
{
    CUVIDSOURCEDATAPACKET packet{};
    packet.payload = packetPointer.data;
    packet.payload_size = packetPointer.size;
    //packet.flags = CUVID_PKT_TIMESTAMP;
    //packet.timestamp = decodedNumber;
    //decodedNumber++;
    if(packetPointer.size == 0)
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
