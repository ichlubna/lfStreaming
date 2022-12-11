#include <memory>
#include <string>
#include <nvidia-sdk/nvcuvid.h>
#include "muxing.h"


#include <iostream>

class VideoDecoder
{
    public:
        VideoDecoder(std::string file);
        ~VideoDecoder();
        void seek(size_t time);
        void initFrame();
        [[nodiscard]]glm::ivec2 getResolution() const
        {
            return demuxer->data.resolution();
        }
        [[nodiscard]] glm::ivec2 getColsRows() const
        {
            return demuxer->data.colsRows();
        }
        friend void operator++(VideoDecoder &decoder)
        {
            decoder.incrementTime();
        }
        void decodeFrame(glm::ivec2 position);
        void flush();
        [[nodiscard]] bool allFramesReady() const;
        void clearBuffer();
        class DecodedFrame
        {
            public:
                CUdeviceptr frame{0};
                unsigned int pitch{0};
                CUvideodecoder decoder{nullptr};
                DecodedFrame(CUvideodecoder dec) : decoder{dec} {};
                DecodedFrame() {};
                ~DecodedFrame()
                {
                    if(frame != 0)
                        cuvidUnmapVideoFrame(decoder, frame);
                }
        };
        [[nodiscard]] const std::vector<DecodedFrame> *getFrames() const
        {
            return &frames;
        };
        [[nodiscard]] const std::vector<void*> getFramePointers() const;

    private:
        static constexpr int DECODED_COUNT{8};
        static constexpr int DECODER_CALLBACK_SUCCESS{1};
        static constexpr int FRAME_COUNT{4};
        size_t decodedNumber{0};
        size_t time{0};
        std::string input;
        std::unique_ptr<Muxing::Demuxer> demuxer;
        CUvideodecoder decoder{nullptr};
        CUvideoparser parser{nullptr};
        void init();
        void checkGPU();
        void createDecoder();
        void createParser();
        void decode(Muxing::Demuxer::PacketPointer packetPointer);
        void incrementTime();
        void prepareFrame(int pictureID, CUVIDPROCPARAMS params);
        std::vector<DecodedFrame> frames;
        cudaVideoChromaFormat chromaFormat{cudaVideoChromaFormat_420};
        cudaVideoCodec getCodec();
        int videoSequence(CUVIDEOFORMAT *format);
        static int CUDAAPI videoSequenceCallback(void *userData, CUVIDEOFORMAT *format)
        {
            return reinterpret_cast<VideoDecoder *>(userData)->videoSequence(format);
        }
        int decodePicture(CUVIDPICPARAMS *picParams);
        static int CUDAAPI decodePictureCallback(void *userData, CUVIDPICPARAMS *picParams)
        {
            return reinterpret_cast<VideoDecoder *>(userData)->decodePicture(picParams);
        }
        int displayPicture(CUVIDPARSERDISPINFO *dispInfo);
        static int CUDAAPI displayPictureCallback(void *userData, CUVIDPARSERDISPINFO *dispInfo)
        {
            return reinterpret_cast<VideoDecoder *>(userData)->displayPicture(dispInfo);
        }

};
