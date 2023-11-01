#include <stdexcept>
#include <iostream>
#include <sstream>
extern "C" {
#include <libavutil/opt.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}
#include "frameComparator.h"

FrameComparator::FrameComparator(const AVCodecContext *sampleCodecContext)
{
    const AVFilter *bufferRef  = avfilter_get_by_name("buffer");
    const AVFilter *bufferTest  = avfilter_get_by_name("buffer");
    //const AVFilter *bufferSinkPsnr = avfilter_get_by_name("buffersink");
    const AVFilter *bufferSinkSsim = avfilter_get_by_name("buffersink");
    //const AVFilter *bufferSinkVmaf = avfilter_get_by_name("buffersink");
    //const AVFilter *psnrFilter  = avfilter_get_by_name("psnr");
    /*if(psnrFilter == nullptr)
        throw std::runtime_error("Cannot find psnr filter.");*/
    const AVFilter *ssimFilter  = avfilter_get_by_name("ssim");
    if(ssimFilter == nullptr)
        throw std::runtime_error("Cannot find ssim filter.");
    /*const AVFilter *vmafFilter  = avfilter_get_by_name("libvmaf");
    if(vmafFilter == nullptr)
        throw std::runtime_error("Cannot find vmaf filter.");*/
    /*const AVFilter *splitFilterRef  = avfilter_get_by_name("split");
    if(splitFilterRef == nullptr)
        throw std::runtime_error("Cannot find split filter.");
    const AVFilter *splitFilterTest  = avfilter_get_by_name("split");
    if(splitFilterTest == nullptr)
        throw std::runtime_error("Cannot find split filter.");*/
    const AVFilter *scaleFilterTest  = avfilter_get_by_name("scale");
    if(scaleFilterTest == nullptr)
        throw std::runtime_error("Cannot find scale filter.");
    const AVFilter *scaleFilterRef  = avfilter_get_by_name("scale");
    if(scaleFilterRef == nullptr)
        throw std::runtime_error("Cannot find scale filter.");

    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE };

    filterGraph = avfilter_graph_alloc();
    if(!filterGraph)
        throw std::runtime_error("Cannot allocate filter graph.");

    std::stringstream arguments;
    arguments << "video_size=" << sampleCodecContext->width << "x" << sampleCodecContext->height <<
              ":pix_fmt=" << sampleCodecContext->pix_fmt <<
              ":time_base=" << /*sampleCodecContext->time_base.num*/ 1 << "/" << sampleCodecContext->time_base.den <<
              ":pixel_aspect=" << sampleCodecContext->sample_aspect_ratio.num << "/" << sampleCodecContext->sample_aspect_ratio.den;
    if(avfilter_graph_create_filter(&bufferRefCtx, bufferRef, "refIn", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph refIn.");
    if(avfilter_graph_create_filter(&bufferTestCtx, bufferTest, "testIn", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph testIn."); 
    arguments.str("1920x1080");
    if(avfilter_graph_create_filter(&scaleFilterRefCtx, scaleFilterRef, "scaleFilter", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph scaleFilterRef.");
    if(avfilter_graph_create_filter(&scaleFilterTestCtx, scaleFilterTest, "scaleFilter", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph scaleFilterTest.");
    /*if(avfilter_graph_create_filter(&bufferSinkPsnrCtx, bufferSinkPsnr, "outPsnr", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph out for psnr.");*/
    if(avfilter_graph_create_filter(&bufferSinkSsimCtx, bufferSinkSsim, "outSsim", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph out for psnr.");
    /*if(avfilter_graph_create_filter(&bufferSinkVmafCtx, bufferSinkVmaf, "outVmaf", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph out for psnr.");
    if(avfilter_graph_create_filter(&psnrFilterCtx, psnrFilter, "psnrFilter", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph psnrFilter.");*/
    if(avfilter_graph_create_filter(&ssimFilterCtx, ssimFilter, "ssimFilter", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph ssimFilter.");
    /*if(avfilter_graph_create_filter(&vmafFilterCtx, vmafFilter, "vmafFilter", nullptr, nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph vmafFilter.");
    arguments.str("3");
    if(avfilter_graph_create_filter(&splitFilterRefCtx, splitFilterRef, "splitFilter", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph splitFilterRef.");
    if(avfilter_graph_create_filter(&splitFilterTestCtx, splitFilterTest, "splitFilter", arguments.str().c_str(), nullptr, filterGraph) < 0)
        throw std::runtime_error("Cannot create filter graph splitFilterTest.");*/

    /*
    if(av_opt_set_int_list(bufferSinkPsnrCtx, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN) < 0)
        throw std::runtime_error("Cannot set pix fmt in graph.");*/
    if(av_opt_set_int_list(bufferSinkSsimCtx, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN) < 0)
        throw std::runtime_error("Cannot set pix fmt in graph.");
    /*if(av_opt_set_int_list(bufferSinkVmafCtx, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN) < 0)
        throw std::runtime_error("Cannot set pix fmt in graph.");*/

    if(avfilter_link(bufferRefCtx, 0, scaleFilterRefCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and scale.");
    if(avfilter_link(bufferTestCtx, 0, scaleFilterTestCtx, 0) != 0)
        throw std::runtime_error("Cannot link test and scale.");

    if(avfilter_link(scaleFilterRefCtx, 0, ssimFilterCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and ssim.");
    if(avfilter_link(scaleFilterTestCtx, 0, ssimFilterCtx, 1) != 0)
        throw std::runtime_error("Cannot link test and ssim.");
    /*
    if(avfilter_link(scaleFilterRefCtx, 0, splitFilterRefCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and split.");
    if(avfilter_link(scaleFilterTestCtx, 0, splitFilterTestCtx, 0) != 0)
        throw std::runtime_error("Cannot link test and split."); 
    if(avfilter_link(splitFilterRefCtx, 0, psnrFilterCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and psnr.");
    if(avfilter_link(splitFilterTestCtx, 0, psnrFilterCtx, 1) != 0)
        throw std::runtime_error("Cannot test reference and psnr.");
    if(avfilter_link(splitFilterRefCtx, 1, ssimFilterCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and ssim.");
    if(avfilter_link(splitFilterTestCtx, 1, ssimFilterCtx, 1) != 0)
        throw std::runtime_error("Cannot link test and ssim.");
    if(avfilter_link(splitFilterRefCtx, 2, vmafFilterCtx, 0) != 0)
        throw std::runtime_error("Cannot link reference and vmaf.");
    if(avfilter_link(splitFilterTestCtx, 2, vmafFilterCtx, 1) != 0)
        throw std::runtime_error("Cannot link test and vmaf.");
    if(avfilter_link(psnrFilterCtx, 0, bufferSinkPsnrCtx, 0) != 0)
        throw std::runtime_error("Cannot link psnr and sink.");
    */
    if(avfilter_link(ssimFilterCtx, 0, bufferSinkSsimCtx, 0) != 0)
        throw std::runtime_error("Cannot link ssim and sink.");
    /*if(avfilter_link(vmafFilterCtx, 0, bufferSinkVmafCtx, 0) != 0)
        throw std::runtime_error("Cannot link vmaf and sink.");*/
    if(avfilter_graph_config(filterGraph, nullptr) < 0)
        throw std::runtime_error("Cannot configure filter graph.");
}

void FrameComparator::printMetadata(AVDictionary *metadata) const
{
    char *entries = nullptr;
    av_dict_get_string(metadata, &entries, ',', '|');
    std::cerr << entries << std::endl;
}

float FrameComparator::getMetric(AVFilterContext *filterContext, std::string dataName)
{
    float metricValue = 0;
    bool waitForFrame{true};
    while(waitForFrame)
    {
        int r = av_buffersink_get_frame(filterContext, resultFrame);
        if(r == AVERROR(EAGAIN) || r == AVERROR_EOF)
            break;
        if(r < 0)
            throw std::runtime_error("Cannot recieve frame from the graph.");
        //printMetadata(resultFrame->metadata);
        auto metricData = av_dict_get(resultFrame->metadata, dataName.c_str(), nullptr, 0)->value;
        metricValue = std::stof(metricData);
        av_frame_unref(resultFrame);
        waitForFrame = false;
    }
    return metricValue;
}

FrameComparator::Metrics FrameComparator::getMetrics()
{
    Metrics metrics;
    //metrics.psnr = getMetric(bufferSinkPsnrCtx, "lavfi.psnr.psnr_avg");
    metrics.ssim = getMetric(bufferSinkSsimCtx, "lavfi.ssim.All");
    //metrics.vmaf = getMetric(bufferSinkVmafCtx, "lavfi.vmaf");
    return metrics;
}

void FrameComparator::pushReference(Frame &frame)
{
    if(av_buffersrc_add_frame_flags(bufferRefCtx, frame.getFrame(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
        throw std::runtime_error("Cannot add reference frame to the graph.");    
}

void FrameComparator::pushDistorted(Frame &frame)
{ 
    if(av_buffersrc_add_frame_flags(bufferTestCtx, frame.getFrame(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
        throw std::runtime_error("Cannot add tested frame to the graph.");
}
