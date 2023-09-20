#include "keyFrameAnalyzer.h"
extern "C" {
#include <libavutil/opt.h> 
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
}
#include <filesystem>

#include <iostream>
KeyFrameAnalyzer::KeyFrameAnalyzer(std::filesystem::path inDirectory) : directory{inDirectory}
{
    const AVFilter *bufferRef  = avfilter_get_by_name("buffer");
    const AVFilter *bufferTest  = avfilter_get_by_name("buffer");
    const AVFilter *bufferSink = avfilter_get_by_name("buffersink");
    const AVFilter *psnrFilter  = avfilter_get_by_name("psnr");
    AVFilterContext *psnrFilterCtx;
    const AVFilter *ssimFilter  = avfilter_get_by_name("ssim");
    AVFilterContext *ssimFilterCtx;
    const AVFilter *vmafFilter  = avfilter_get_by_name("libvmaf");
    AVFilterContext *vmafFilterCtx;
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE };

    filterGraph = avfilter_graph_alloc();
    if (!filterGraph) 
        throw std::runtime_error("Cannot allocate filter graph.");

    std::filesystem::directory_iterator files(directory); 
        Frame sampleFrame((*files).path().string());
    auto sampleCodecContext = sampleFrame.getCodecContext();
   
    std::stringstream arguments; 
    arguments << "video_size=" << sampleCodecContext->width << "x" << sampleCodecContext->height <<
    ":pix_fmt=" << sampleCodecContext->pix_fmt << 
    ":time_base=" << /*sampleCodecContext->time_base.num*/ 1 << "/" << sampleCodecContext->time_base.den <<
    ":pixel_aspect=" << sampleCodecContext->sample_aspect_ratio.num << "/" << sampleCodecContext->sample_aspect_ratio.den; 
    if(avfilter_graph_create_filter(&bufferRefCtx, bufferRef, "refIn", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph refIn.");
    if(avfilter_graph_create_filter(&bufferTestCtx, bufferTest, "testIn", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph testIn.");
    if(avfilter_graph_create_filter(&bufferSinkCtx, bufferSink, "out", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph out.");
    if(avfilter_graph_create_filter(&psnrFilterCtx, psnrFilter, "psnrFilter", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph psnrFilter.");
    //if(avfilter_graph_create_filter(&ssimFilterCtx, ssimFilter, "ssimFilter", args, nullptr, filterGraph) < 0)
    //    throw std::runtime_error("Cannot create filter graph ssimFilter.");
    //if(avfilter_graph_create_filter(&vmafFilterCtx, vmafFilter, "vmafFilter", args, nullptr, filterGraph) < 0)
    //    throw std::runtime_error("Cannot create filter graph vmafFilter.");


    if(av_opt_set_int_list(bufferSinkCtx, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN)<0)
        throw std::runtime_error("Cannot set pix fmt in graph.");

    avfilter_link(bufferRefCtx, 0, psnrFilterCtx, 0);
    avfilter_link(bufferTestCtx, 0, psnrFilterCtx, 1);
    avfilter_link(psnrFilterCtx, 0, bufferSinkCtx, 0);
    if (avfilter_graph_config(filterGraph, nullptr) < 0)
        throw std::runtime_error("Cannot configure filter graph.");
}

std::filesystem::path KeyFrameAnalyzer::getBestKeyFrame()
{
    BestMetrics bestMetrics;

    std::filesystem::directory_iterator files(directory); 
    for (const auto &candidateFile : files)
    {
        Frame candidateFrame(candidateFile.path());
        for (const auto &testedFile : files)
        {
            Frame testedFrame(testedFile.path());
            if(av_buffersrc_add_frame(bufferRefCtx, candidateFrame.getFrame()) < 0)  
                throw std::runtime_error("Cannot add reference frame to the graph.");
            if(av_buffersrc_add_frame(bufferTestCtx, testedFrame.getFrame()) < 0)  
                throw std::runtime_error("Cannot add tested frame to the graph.");
           
            AVFrame *resultFrame = av_frame_alloc(); 
            bool waitForFrame{true};
            while (waitForFrame) 
            {
                int r = av_buffersink_get_frame(bufferSinkCtx, resultFrame);
                if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
                    break;
                if (r < 0)
                    throw std::runtime_error("Cannot recieve frame from the graph.");
                //process
                std::cerr << "aaa:" << resultFrame->format;
                exit(0);
                av_frame_unref(resultFrame);
                waitForFrame = false;
            }
        }
    }
}

