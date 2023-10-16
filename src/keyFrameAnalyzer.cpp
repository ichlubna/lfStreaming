#include <stdexcept>
#include <filesystem>
#include "keyFrameAnalyzer.h"
#include "frameComparator.h"
#include "loadingBar/loadingbar.hpp"

KeyFrameAnalyzer::KeyFrameAnalyzer(std::filesystem::path inDirectory) : directory {inDirectory}
{
}

std::filesystem::path KeyFrameAnalyzer::getBestKeyFrame()
{ 
    size_t count = std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{});
    LoadingBar bar(count*count, true);
    BestMetrics bestMetrics;

    std::filesystem::directory_iterator files(directory);
    Frame sampleFrame((*files).path().string());
    auto sampleCodecContext = sampleFrame.getCodecContext(); 
    FrameComparator comparator(sampleCodecContext);   
 
    for(const auto &candidateFile : std::filesystem::directory_iterator{directory})
    {
        Frame candidateFrame(candidateFile.path());
        bestMetrics.newCandidate(candidateFile.path());
        for(const auto &testedFile : std::filesystem::directory_iterator{directory})
        {
            comparator.pushReference(candidateFrame);
            Frame testedFrame(testedFile.path());
            comparator.pushDistorted(testedFrame);
            FrameComparator::Metrics metrics = comparator.getMetrics();
            bestMetrics.add(metrics.psnr, metrics.ssim, metrics.vmaf);
            std::cerr << metrics.psnr << " " << metrics.ssim << " " << metrics.vmaf << std::endl;
            bar.add();
        }
    }
    return bestMetrics.result();
}
