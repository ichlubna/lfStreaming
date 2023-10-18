#include <filesystem>
#include <vector>
#include "muxing.h"

class KeyFrameAnalyzer
{
    public:
        [[nodiscard]] std::filesystem::path getBestKeyFrame(std::filesystem::path directory);
        [[nodiscard]] bool isSignificantlyDifferent(std::vector<std::filesystem::path> listA, std::vector<std::filesystem::path> listB);

    private:
        class BestMetrics
        {
            public:
                class MetricResult
                {
                    private: 
                        float value{0};
                        size_t count{0};
                        std::filesystem::path path{""};
                    public:
                        MetricResult(float startValue=0) : value{startValue}
                        {
                        }
                        void clear(std::filesystem::path newPath = "")
                        {
                            value = 0;
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
                MetricResult bestPsnr{FLT_MAX};
                MetricResult bestSsim{FLT_MAX};
                MetricResult bestVmaf{FLT_MAX};
                MetricResult currentPsnr;
                MetricResult currentSsim;
                MetricResult currentVmaf; 
        };
        class Comparator
        {
            public:
            glm::vec2 reference;
            Comparator(glm::uvec2 referenceCoords) : reference{referenceCoords}{};
            bool operator () (std::filesystem::path a, std::filesystem::path b)
            {
                return glm::distance(reference, glm::vec2(Muxing::parseFilename(a))) > glm::distance(reference, glm::vec2(Muxing::parseFilename(b)));
            } 
        };

        std::vector<std::filesystem::path> selectFrames(std::filesystem::path directory, glm::ivec2 reference, int selectedCount);
        std::filesystem::path directory;
};
