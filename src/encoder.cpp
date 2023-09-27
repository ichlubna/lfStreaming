extern "C" {
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <stdexcept>
#include <filesystem>
#include "encoder.h"
#include "libs/loadingBar/loadingbar.hpp"

const std::vector<uint8_t> Encoder::extractPacketData(AVPacket *packet) const
{
    return std::vector<uint8_t>(&packet->data[0], &packet->data[packet->size]);
}

size_t Encoder::calculateCrf(StreamFormat format, float quality) const
{
    constexpr size_t MAX_H265_CRF{51};
    constexpr size_t MAX_AV1_CRF{63};

    if(quality < 0 || quality > 1)
        throw std::runtime_error("The quality parameter should be in <0;1> interval.");
    float inverseQuality = 1.0 - quality;
    if(format == StreamFormat::H265)
        return round(inverseQuality * MAX_H265_CRF);
    else //if (format == StreamFormat::AV1)
        return round(inverseQuality * MAX_AV1_CRF);
}

Encoder::StreamFormat Encoder::stringToFormat(std::string format) const
{
    if(format == "H265")
        return StreamFormat::H265;
    else if(format == "AV1")
        return StreamFormat::AV1;
    else
        throw std::runtime_error("The specified video stream format is not supported.");
}

void Encoder::checkDir(std::string path) const
{
    if(!std::filesystem::exists(path))
        throw std::runtime_error("The path " + path + " does not exist!");
    if(!std::filesystem::is_directory(path))
        throw std::runtime_error("The path " + path + " does not lead to a directory!");
    if(std::filesystem::is_empty(path))
        throw std::runtime_error("The directory " + path + " is empty!");
}

void Encoder::encode(std::string inputDir, std::string outputFile, float quality, std::string format, glm::ivec2 keyCoords, int keyInterval, float aspect, glm::vec2 focusRange)
{
    if(keyInterval < 1)
        throw std::runtime_error("The GoP interval can't be lower than 1!");
    checkDir(inputDir);
    auto timeFrameDirs = Muxing::listPath(inputDir);
    timeFrameCount = timeFrameDirs.size();
    for(auto const &dir : timeFrameDirs)
        encodeTimeFrame(inputDir / dir, quality, format, aspect, focusRange, keyCoords, keyInterval);
    muxer->save(outputFile);
}

void Encoder::encodeTimeFrame(std::string inputDir, float quality, std::string format, float aspect, glm::vec2 focusRange, glm::ivec2 keyCoords, int keyInterval)
{
    checkDir(inputDir);
    auto files = Muxing::listPath(inputDir);
    auto lastFileCoords = Muxing::parseFilename(*files.rbegin()) + glm::uvec2(1);
    auto colsRows = lastFileCoords;
    auto referenceCoords = keyCoords;
    if(keyCoords == glm::ivec2(-1, -1))
        referenceCoords = lastFileCoords / glm::uvec2(2);
    auto videoFormat = stringToFormat(format);
    size_t crf = calculateCrf(videoFormat, quality);

    std::cout << "Time frame " << currentFrame + 1 << " of " << timeFrameCount << std::endl;
    std::cout << "Encoding..." << std::endl;
    LoadingBar bar(files.size() + 1, true);

    std::filesystem::path path{inputDir};
    if((currentFrame % keyInterval) == 0)
    {
        KeyFrameAnalyzer keyFrameAnalyzer(inputDir);
        keyFrameAnalyzer.getBestKeyFrame();
        
        size_t referenceIndex = referenceCoords.y * colsRows.x + referenceCoords.x;
        std::set<std::filesystem::path>::iterator it = files.begin();
        std::advance(it, referenceIndex);
        std::string referenceFrame = *it;
        lastReferenceFrame.fileName = path / referenceFrame;
        lastReferenceFrame.timeFrame = currentFrame;
        lastReferenceFrame.coords = referenceCoords;
    }
    PairEncoder refFrame(lastReferenceFrame.fileName, lastReferenceFrame.fileName, crf, videoFormat);
    auto resolution = refFrame.getResolution();

    if(!muxer->isInitialized())
        muxer->init(resolution, colsRows, videoFormat, timeFrameCount, aspect, focusRange);

    for(auto const &file : files)
        if(lastReferenceFrame.fileName == path / file)
        {
            *muxer << refFrame.getReferencePacket();
            bar.add();
        }
        else
        {
            PairEncoder newFrame(lastReferenceFrame.fileName, path / file, crf, videoFormat);
            bar.add();
            *muxer << newFrame.getFramePacket();
        }
    muxer->endTimeFrame(lastReferenceFrame.coords, lastReferenceFrame.timeFrame);
    currentFrame++;
    bar.add();
}

void Encoder::FFEncoder::initH265()
{
    codecName = "libx265";
    codecParamsName = "x265-params";
    codecParams = "log-level=error:keyint=" + std::to_string(NO_KEYINT) + ":min-keyint=" + std::to_string(NO_KEYINT) + ":scenecut=0:crf=" + std::to_string(crf);
}

void Encoder::FFEncoder::initAV1()
{
    codecName = "libaom-av1";
    codecParamsName = "aom-params";
    codecParams = "keyint_min=" + std::to_string(NO_KEYINT) + ":crf=" + std::to_string(crf);
}

Encoder::FFEncoder::FFEncoder(glm::uvec2 inResolution, AVPixelFormat inPixFmt, size_t inCrf, Encoder::StreamFormat format) : resolution{inResolution}, pixFmt{inPixFmt}, crf{inCrf}
{
    if(format == Encoder::StreamFormat::H265)
        initH265();
    else if(format == Encoder::StreamFormat::AV1)
        initAV1();
    initEncoder();
}

void Encoder::FFEncoder::initEncoder()
{
    codec = avcodec_find_encoder_by_name(codecName.c_str());
    codecContext = avcodec_alloc_context3(codec);
    if(!codecContext)
        throw std::runtime_error("Cannot allocate output context!");
    codecContext->width = resolution.x;
    codecContext->height = resolution.y;
    codecContext->pix_fmt = pixFmt;
    codecContext->time_base = {1, 60};
    av_opt_set(codecContext->priv_data, codecParamsName.c_str(), codecParams.c_str(), 0);
    if(avcodec_open2(codecContext, codec, nullptr) < 0)
        throw std::runtime_error("Cannot open output codec!");
    packet = av_packet_alloc();
}

Encoder::FFEncoder::~FFEncoder()
{
    avcodec_free_context(&codecContext);
    av_packet_free(&packet);
}

AVPacket *Encoder::FFEncoder::retrievePacket()
{
    bool waitForPacket = true;
    while(waitForPacket)
    {
        int err = avcodec_receive_packet(codecContext, packet);
        if(err == AVERROR_EOF || err == AVERROR(EAGAIN))
            return nullptr;
        else if(err < 0)
            throw std::runtime_error("Cannot receive packet");
        waitForPacket = false;
    }
    return packet;
}

void Encoder::FFEncoder::encodeFrame(AVFrame *frame)
{
    avcodec_send_frame(codecContext, frame);
}

AVFrame *Encoder::PairEncoder::convertFrame(const AVFrame *inputFrame, AVPixelFormat pxFormat) const
{
    constexpr int CHANNELS{3};
    AVFrame *outputFrame = av_frame_alloc();
    outputFrame->width = inputFrame->width;
    outputFrame->height = inputFrame->height;
    outputFrame->format = pxFormat;
    av_frame_get_buffer(outputFrame, CHANNELS * 8);
    auto swsContext = sws_getContext(inputFrame->width, inputFrame->height, static_cast<AVPixelFormat>(inputFrame->format),
                                     inputFrame->width, inputFrame->height, pxFormat, SWS_BICUBIC, nullptr, nullptr, nullptr);
    if(!swsContext)
        throw std::runtime_error("Cannot get conversion context!");
    sws_scale(swsContext, inputFrame->data, inputFrame->linesize, 0, inputFrame->height, outputFrame->data, outputFrame->linesize);
    return outputFrame;
}

void Encoder::PairEncoder::encode()
{
    Frame reference(referenceFile);
    Frame frame(frameFile);
    auto referenceFrame = reference.getFrame();

    width = referenceFrame->width;
    height = referenceFrame->height;

    FFEncoder encoder({referenceFrame->width, referenceFrame->height}, outputPixelFormat, crf, format);

    auto convertedReference = convertFrame(reference.getFrame(), outputPixelFormat);
    auto convertedFrame = convertFrame(frame.getFrame(), outputPixelFormat);

    encoder << convertedReference;
    AVFrame *convertedFrameRaw = convertedFrame;
    convertedFrameRaw->key_frame = 0;
    encoder << convertedFrameRaw;
    encoder << nullptr;

    std::vector<uint8_t> *buffer = &referencePacket;
    AVPacket *packet;
    for(int i = 0; i < 2; i++)
    {
        encoder >> &packet;
        if(!packet)
            throw std::runtime_error("Cannot receieve packet!");
        buffer->insert(buffer->end(), &packet->data[0], &packet->data[packet->size]);
        buffer = &framePacket;
    }
}
