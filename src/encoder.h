#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <set>
#include <memory>
#include "muxing.h"
#include "frame.h"
#include "keyFrameAnalyzer.h"

class Encoder
{
    public:
        Encoder() : muxer{std::make_unique<Muxing::Muxer>()} {};
        void encode(std::string inputDir, std::string outputFile, float quality, std::string format, glm::ivec2 keyCoords, int keyInterval, float aspect, glm::vec2 focusRange);
        const std::vector<uint8_t> extractPacketData(AVPacket *packet) const;
        static const AVPixelFormat outputPixelFormat{AV_PIX_FMT_YUV420P};
        enum StreamFormat { H265 = 0, AV1 = 1 };

    private:
        class FFEncoder
        {
            public:
                FFEncoder(glm::uvec2 inResolution, AVPixelFormat inPixFmt, size_t inCrf, Encoder::StreamFormat format);
                ~FFEncoder();
                friend void operator<<(FFEncoder &e, AVFrame *frame)
                {
                    e.encodeFrame(frame);
                }
                friend void operator>>(FFEncoder &e, AVPacket **packetPtr)
                {
                    *packetPtr = e.retrievePacket();
                }
                const AVCodecContext *getCodecContext() const
                {
                    return codecContext;
                };

            private:
                void encodeFrame(AVFrame *frame);
                void initH265();
                void initAV1();
                void initEncoder();
                glm::uvec2 resolution;
                AVPixelFormat pixFmt;
                size_t crf;
                std::string codecName;
                std::string codecParamsName;
                std::string codecParams;
                AVPacket *retrievePacket();
                const AVCodec *codec;
                AVStream *stream;
                AVCodecContext *codecContext;
                AVPacket *packet;
                static constexpr size_t NO_KEYINT{99999};
        };

        class PairEncoder
        {
            public:
                PairEncoder(std::string ref, std::string frame, size_t inCrf, Encoder::StreamFormat inFormat) : referenceFile(ref), frameFile(frame), crf{inCrf}, format{inFormat}
                {
                    encode();
                };
                [[nodiscard]] const std::vector<uint8_t> *getFramePacket() const
                {
                    return &framePacket;
                };
                [[nodiscard]] const std::vector<uint8_t> *getReferencePacket() const
                {
                    return &referencePacket;
                };
                [[nodiscard]] const glm::uvec2 getResolution() const
                {
                    return {width, height};
                };

            private:
                [[nodiscard]] AVFrame *convertFrame(const AVFrame *inputFrame, AVPixelFormat pxFormat) const;
                void encode();
                std::string referenceFile;
                std::string frameFile;
                std::vector<uint8_t> framePacket;
                std::vector<uint8_t> referencePacket;
                size_t crf;
                size_t width;
                size_t height;
                Encoder::StreamFormat format;
        };

        class LastReferenceFrame
        {
            public:
                std::filesystem::path fileName;
                glm::uvec2 coords;
                size_t timeFrame;
        } lastReferenceFrame;

        size_t calculateCrf(StreamFormat format, float quality) const;
        size_t timeFrameCount{0};
        StreamFormat stringToFormat(std::string) const;
        void encodeTimeFrame(std::string inputDir, float quality, std::string format, float aspect, glm::vec2 focusRange, glm::ivec2 keyCoords, int keyInterval);
        void checkDir(std::string path) const;
        std::unique_ptr<Muxing::Muxer> muxer;
        size_t currentFrame{0};
};
