#include <filesystem>
#include <vector>
#include "muxing.h"

#include <iostream>
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
                        MetricResult(std::filesystem::path inPath="") : path{inPath}
                        {
                        }
                        [[nodiscard]] float average()
                        {
                            return value/count;
                        }
                        MetricResult& operator+=(const float& rhs)
                        {
                            this->value += rhs;
                            this->count++;
                            return *this;
                        }
                        [[nodiscard]] static MetricResult& max(MetricResult &a, MetricResult &b)
                        {
                            if(a.average() > b.average())
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
                    currentSsim += inSsim;
                    currentVmaf += inVmaf; 
                }

                void newCandidate(std::filesystem::path path)
                {    
                    bestPsnr = MetricResult::max(bestPsnr, currentPsnr); 
                    bestSsim = MetricResult::max(bestSsim, currentSsim); 
                    bestVmaf = MetricResult::max(bestVmaf, currentVmaf);
                    currentPsnr = MetricResult(path); 
                    currentSsim = MetricResult(path); 
                    currentVmaf = MetricResult(path);
                } 

                [[nodiscard]] const std::filesystem::path result()
                {
                    return bestSsim.result();
                }
            private:
                MetricResult bestPsnr;
                MetricResult bestSsim;
                MetricResult bestVmaf;
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
