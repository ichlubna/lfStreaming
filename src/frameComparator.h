#include "frame.h"
extern "C" {
#include <libavfilter/avfilter.h>
}

class FrameComparator
{
    public:
        class Metrics
        {
            public:
            float psnr, ssim, vmaf;
        };

        FrameComparator(const AVCodecContext *sampleCodecContext);
        ~FrameComparator()
        {
            av_frame_free(&resultFrame);
        }
        void pushReference(Frame &frame);
        void pushDistorted(Frame &frame);
        [[nodiscard]] Metrics getMetrics();

    private:
        AVFilterGraph *filterGraph;
        AVFilterContext *bufferSinkPsnrCtx;
        AVFilterContext *bufferSinkSsimCtx;
        AVFilterContext *bufferSinkVmafCtx;
        AVFilterContext *bufferRefCtx;
        AVFilterContext *bufferTestCtx;
        AVFrame *resultFrame = av_frame_alloc();
        void printMetadata(AVDictionary *metadata) const;
            float getMetric(AVFilterContext  *filterContext, std::string dataName);
};
