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
            public:
                class MetricResult
                {
                    private: 
                        float value{0};
                        size_t count{0};
                        std::filesystem::path path;
                    public:
                        MetricResult()
                        {
                            clear(); 
                        }
                        void clear(std::filesystem::path newPath = "")
                        {
                            value = 999999;
                            count = 0;
                            path = newPath;
                        }
                        [[nodiscard]] float average()
                        {
                            return value/count;
                        }
                        MetricResult& operator+=(const float& rhs)
                        {
                            value += rhs;
                            count++;
                            return *this;
                        }
                        [[nodiscard]] static MetricResult& min(MetricResult &a, MetricResult &b)
                        {
                            if(a.average() < b.average())
                                return a;
                            return b;
                        }
                        [[nodiscard]] const std::filesystem::path result()
                        {
                            return path;
                        }
                   };

                void add(float inPsnr, float inSsim, float inVmaf)
                {
                    currentPsnr += inPsnr;
                    currentPsnr += inSsim;
                    currentPsnr += inVmaf; 
                }

                void newCandidate(std::filesystem::path path)
                {    
                    bestPsnr = MetricResult::min(bestPsnr, currentPsnr); 
                    bestSsim = MetricResult::min(bestSsim, currentSsim); 
                    bestVmaf = MetricResult::min(bestVmaf, currentVmaf);
                    currentPsnr.clear(path); 
                    currentSsim.clear(path); 
                    currentVmaf.clear(path);
                }

                [[nodiscard]] const std::filesystem::path result()
                {
                    return bestPsnr.result();
                }
            private:
                MetricResult bestPsnr;
                MetricResult bestSsim;
                MetricResult bestVmaf;
                MetricResult currentPsnr;
                MetricResult currentSsim;
                MetricResult currentVmaf; 


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
