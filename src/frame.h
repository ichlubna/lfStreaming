extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include <string>

class Frame
{
    public:
        Frame(std::string file);
        ~Frame();
        AVFrame *getFrame() const
        {
            return frame;
        };

        const AVCodecContext* getCodecContext() const
        {
            return codecContext;
        }

    private:
        AVFormatContext *formatContext;
        AVCodec *codec;
        AVStream *stream;
        AVCodecContext *codecContext;
        AVFrame *frame;
        AVPacket *packet;
    
};
