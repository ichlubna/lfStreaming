#include <filesystem>
#include "frame.h"
extern "C" {
#include <libavfilter/avfilter.h>
}

class KeyFrameAnalyzer
{
    public:
        KeyFrameAnalyzer(std::filesystem::path directory);
        std::filesystem::path getBestKeyFrame();
        ~KeyFrameAnalyzer()
        {
            av_frame_free(&resultFrame);
        }

    private:
        class BestMetrics
        {
            private:

            public:
                class MetricResult
                {
                    public:
                        std::filesystem::path path;
                        float value;
                };
                MetricResult psnr;
                MetricResult ssim;
                MetricResult vmaf;
        };
        AVFilterGraph *filterGraph;
        AVFilterContext *bufferSinkPsnrCtx;
        AVFilterContext *bufferSinkSsimCtx;
        AVFilterContext *bufferSinkVmafCtx;
        AVFilterContext *bufferRefCtx;
        AVFilterContext *bufferTestCtx;
        AVFrame *resultFrame = av_frame_alloc();
        std::filesystem::path directory;
        [[nodiscard]] float getMetric(std::string dataName, AVFilterContext *filterContext);
        void printMetadata(AVDictionary *metadata) const;
};
