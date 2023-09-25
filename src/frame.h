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
        [[nodiscard]] AVFrame *getFrame() const
        {
            return frame;
        };

        [[nodiscard]] const AVCodecContext *getCodecContext() const
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
