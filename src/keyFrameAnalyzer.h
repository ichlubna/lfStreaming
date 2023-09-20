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
    AVFilterContext *bufferSinkCtx;
    AVFilterContext *bufferRefCtx; 
    AVFilterContext *bufferTestCtx;
    std::filesystem::path directory; 
};
