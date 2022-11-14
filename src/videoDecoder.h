#include <memory>
#include <string>
#include <nvidia-sdk/nvcuvid.h>
#include "muxing.h"

class VideoDecoder
{
    public:
    VideoDecoder(std::string file);

    private:
    std::string input;
    std::unique_ptr<Muxing::Demuxer> demuxer;
    CUvideodecoder decoder{nullptr};
    void init();
    void checkGPU();
    void createDecoder();
    cudaVideoChromaFormat chromaFormat{cudaVideoChromaFormat_420};
    cudaVideoCodec getCodec();
};
