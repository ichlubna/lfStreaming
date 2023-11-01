#include <stdexcept>
#include <filesystem>
#include <list>
#include <bits/stdc++.h> 
#include "keyFrameAnalyzer.h"
#include "frameComparator.h"
#include "loadingBar/loadingbar.hpp"

bool KeyFrameAnalyzer::isSignificantlyDifferent(std::vector<std::filesystem::path> listA, std::vector<std::filesystem::path> listB)
{
    Frame sampleFrame(listA.front());
    auto sampleCodecContext = sampleFrame.getCodecContext(); 
    FrameComparator comparator(sampleCodecContext);
    FrameComparator::Metrics result;
    for(size_t i=0; i<listA.size(); i++)
    {
        Frame a(listA[i]);
        comparator.pushReference(a);
        Frame b(listB[i]);
        comparator.pushReference(b);
        FrameComparator::Metrics metrics = comparator.getMetrics();
        result += {metrics.psnr, metrics.ssim, metrics.vmaf};
    }
    auto average = result.average();
    constexpr float PSNR_LIMIT{25};
    constexpr float SSIM_LIMIT{0.9};
    constexpr float VMAF_LIMIT{60};
    
    return average.psnr < PSNR_LIMIT && average.ssim < SSIM_LIMIT && average.vmaf < VMAF_LIMIT;
}

std::vector<std::filesystem::path> KeyFrameAnalyzer::selectFrames(std::filesystem::path directory, glm::ivec2 reference, int selectedCount)
{
    std::vector<std::filesystem::path> files; 
    for(const auto &testedFile : std::filesystem::directory_iterator{directory})
        files.push_back(testedFile);
    std::sort(files.begin(), files.end(), Comparator(reference));
    int count = files.size();
    if(selectedCount > 0 && selectedCount < count)
        count = selectedCount;
    files.resize(count);
    return files;
}

std::filesystem::path KeyFrameAnalyzer::getBestKeyFrame(std::filesystem::path directory)
{ 
    size_t count = std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{});
    size_t sampleCount = 4;
    LoadingBar bar(count*sampleCount, true);
    BestMetrics bestMetrics;

    std::filesystem::directory_iterator files(directory);
    Frame sampleFrame((*files).path().string());
    auto sampleCodecContext = sampleFrame.getCodecContext(); 
    FrameComparator comparator(sampleCodecContext);   
 
    for(const auto &candidateFile : std::filesystem::directory_iterator{directory})
    {
        Frame candidateFrame(candidateFile.path());
        bestMetrics.newCandidate(candidateFile.path());
        auto files = selectFrames(directory, Muxing::parseFilename(candidateFile.path()),sampleCount);  
        for(const auto &testedFile : files)
        {
            if(candidateFile.path() != testedFile)
            {
                comparator.pushReference(candidateFrame);
                Frame testedFrame(testedFile);
                comparator.pushDistorted(testedFrame);
                FrameComparator::Metrics metrics = comparator.getMetrics();
                bestMetrics.add(metrics.psnr, metrics.ssim, metrics.vmaf);
                //std::cerr << metrics.psnr << " " << metrics.ssim << " " << metrics.vmaf << candidateFile << " " << testedFile << std::endl;
            }
            bar.add();
        }
    }
    return bestMetrics.result();
}

