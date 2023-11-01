#include "frame.h"
extern "C" {
#include <libavfilter/avfilter.h>
}
#include <iostream>
class FrameComparator
{
    public:
        class Metrics
        {
            public:
            float psnr{0}, ssim{0}, vmaf{0};
            int addCount{0};
            Metrics average()
            {
                return {psnr/addCount, ssim/addCount, vmaf/addCount};
            }
            Metrics& operator+=(const Metrics& rhs)
            {
                this->psnr += rhs.psnr;
                this->ssim += rhs.ssim;
                this->vmaf += rhs.vmaf;
                addCount++;
                return *this;
            }
        };

        FrameComparator(const AVCodecContext *sampleCodecContext);
        ~FrameComparator()
        {
            av_frame_free(&resultFrame);
            avfilter_graph_free(&filterGraph);
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
        AVFilterContext *psnrFilterCtx;
        AVFilterContext *ssimFilterCtx;
        AVFilterContext *vmafFilterCtx;
        AVFilterContext *splitFilterRefCtx;
        AVFilterContext *splitFilterTestCtx;
        AVFilterContext *scaleFilterRefCtx;
        AVFilterContext *scaleFilterTestCtx;
        AVFrame *resultFrame = av_frame_alloc();
        void printMetadata(AVDictionary *metadata) const;
            float getMetric(AVFilterContext  *filterContext, std::string dataName);
};
