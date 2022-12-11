#include <chrono>
#include <cuda_runtime.h>

template<bool CPU, bool GPU>
class Timer
{
    public:
    Timer()
    {
        start();
    }
    
    class Times
    {
        public:
        float gpuTime, cpuTime;
    }; 
    
    void start()
    {
        if constexpr (CPU)
            startCPU();    
        if constexpr (GPU)
            startGPU();
    } 
 
    [[nodiscard]] const Times stop()
    {
        Times times;
        if constexpr (CPU)
            times.cpuTime = stopCPU();    
        if constexpr (GPU)
            times.gpuTime = stopGPU();
    } 
    
    private:
    cudaEvent_t startEvent, stopEvent;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    void startGPU()
    { 
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent);
    }

    void startCPU()
    {
        startTime = std::chrono::high_resolution_clock::now();
    }

    [[nodiscard]] float stopCPU()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count(); 
    }

    [[nodiscard]] float stopGPU()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float time = 0;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return time; 
    };
};
