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
    glm::ivec2 getResolution() {return demuxer->data.resolution();}
    friend void operator++(VideoDecoder &decoder){decoder.incrementTime();}
    void decodeFrame(glm::ivec2 position);
    class FramePair
    {
        public:
        CUdeviceptr frames[2]{0,0};
        int pitch{0};
        CUdeviceptr operator[](int index){return frames[index];}
    };
    FramePair getFrames();

    private:
    template <typename T>
    class RingBuffer
    {
        public:
        RingBuffer(size_t size) : data{std::vector<T>(size)}{}
        T* add(T element) {data[end]=element; T* top = &data[end]; end++; end%=data.size(); return top;}; 
        T operator[](int index){return data[index];}
        void clear() {data.clear();}
        
        private:
        size_t end{0};
        std::vector<T> data;
    };

    class DecodedFrame
    {
        public:
        CUdeviceptr frame{0};
        unsigned int pitch{0};
        size_t index{0};
        CUvideodecoder decoder{nullptr};
        DecodedFrame(CUvideodecoder dec) : decoder{dec}{};
        DecodedFrame(){};
        ~DecodedFrame(){if(frame) cuvidUnmapVideoFrame(decoder, frame);}
    };

    static constexpr int DECODED_COUNT{8};
    static constexpr int DECODER_CALLBACK_SUCCESS{1};
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
    RingBuffer<DecodedFrame> frames{2};
    cudaVideoChromaFormat chromaFormat{cudaVideoChromaFormat_420};
    cudaVideoCodec getCodec();
    int videoSequence(CUVIDEOFORMAT *format); 
    static int CUDAAPI videoSequenceCallback(void *userData, CUVIDEOFORMAT *format) { return reinterpret_cast<VideoDecoder*>(userData)->videoSequence(format); }
    int decodePicture(CUVIDPICPARAMS *picParams); 
    static int CUDAAPI decodePictureCallback(void *userData, CUVIDPICPARAMS *picParams) { return reinterpret_cast<VideoDecoder*>(userData)->decodePicture(picParams); }
    int displayPicture(CUVIDPARSERDISPINFO *dispInfo); 
    static int CUDAAPI displayPictureCallback(void *userData, CUVIDPARSERDISPINFO *dispInfo) { return reinterpret_cast<VideoDecoder*>(userData)->displayPicture(dispInfo); }

};
