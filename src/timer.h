#include <chrono>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

template<bool CPU, bool GPU>
class Timer
{
    public:
    Timer()
    {
    }
    
    template<bool CPUTime, bool GPUTime>
    class Times
    {
        public:
        float gpuTime, cpuTime;
        void print(std::string type, float value) const
        {
            std::cout << type << " elapsed time: " << value << " ms" << std::endl;
        }
        void printElapsedGPU() const
        {
            print("GPU", gpuTime);
        }
        void printElapsedCPU() const
        {
            print("CPU", cpuTime);
        }
        void printElapsed() const
        {
            if constexpr (CPUTime)
                printElapsedCPU();
            if constexpr (CPUTime)
                printElapsedGPU();
        }
    }; 
    
    void start()
    {
        if constexpr (CPU)
            startCPU();    
        if constexpr (GPU)
            startGPU();
    } 
 
    [[nodiscard]] const Times<CPU, GPU> stop()
    {
        Times<CPU, GPU> times;
        if constexpr (CPU)
            times.cpuTime = stopCPU();    
        if constexpr (GPU)
            times.gpuTime = stopGPU();
        return times;
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
