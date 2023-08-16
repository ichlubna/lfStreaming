#include <stdint.h>
#include <stdio.h>
#include "kernels.h"

__device__ int linearCoords(int2 coords, int2 resolution)
{
    int linearCoords = coords.y * resolution.x + coords.x;
    return linearCoords;
}
__device__ bool coordsOutside(int2 coords, int2 resolution)
{
    return (coords.x >= resolution.x || coords.y >= resolution.y);
}

__device__ int2 getImgCoords()
{
    int2 coords;
    coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
    coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
    return coords;
}

namespace PerPixelInterpolation
{
    namespace Inputs
    {
        __constant__ float weights[INPUT_COUNT];
        __constant__ int pitches[INPUT_COUNT];
        __constant__ size_t framesY[INPUT_COUNT];
        __constant__ size_t framesUV[INPUT_COUNT];
        __constant__ float2 offsets[INPUT_COUNT];
        __constant__ int pixelCounts[INPUT_COUNT];
        __constant__ uint8_t* resultY;
        __constant__ uint8_t* resultUV;
        __constant__ float inverseWeightSum;
        __constant__ int2 resolution;
        __constant__ int2 doubleResolution;
        __constant__ int2 resolutionMinusOne;
        __constant__ int pitch;
        __constant__ int pixelCount;
        __constant__ float2 focusRange;
        constexpr int FOCUS_STEPS{16};
        __constant__ float focusStep;
    }
constexpr int KERNEL{2}; 
constexpr int KERNEL_WIDTH{KERNEL*2+1};

__device__ uint8_t loadY(int frameID, int2 coords)
{
    int linear = linearCoords(coords, {Inputs::pitches[frameID], Inputs::resolution.y});
    return reinterpret_cast<uint8_t*>(Inputs::framesY[frameID])[linear];
}

__device__ uint3 loadClosestY(int frameID, int2 coords)
{  
    constexpr int UNIT_SIZE{4}; 
    int linear = linearCoords(coords, {Inputs::pitches[frameID], Inputs::resolution.y}) - KERNEL;
    if(linear<0)
        linear=0;
    int roundLinear = linear/UNIT_SIZE;
    uint offset = linear-roundLinear*UNIT_SIZE;
    uint2 sample;
    sample.x = reinterpret_cast<uint*>(Inputs::framesY[frameID])[roundLinear];
    sample.y = reinterpret_cast<uint*>(Inputs::framesY[frameID])[roundLinear+1];
    return {sample.x, sample.y, offset};
}

__device__ uchar2 loadUV(int frameID, int2 coords)
{
    int linear = linearCoords({coords.x-(coords.x&1), coords.y>>1}, {Inputs::pitches[frameID], Inputs::resolution.y});
    uint8_t *UVPlane = reinterpret_cast<uint8_t*>(Inputs::framesY[frameID])+Inputs::pixelCounts[frameID];
    return {UVPlane[linear], UVPlane[linear+1]}; 
}

__device__ void store(uchar3 yuv, int2 coords)
{
    int linear = linearCoords(coords, {Inputs::pitch, Inputs::resolution.y});
    Inputs::resultY[linear] = yuv.x;
    linear = linearCoords({coords.x-(coords.x&1), coords.y>>1}, {Inputs::pitch, Inputs::resolution.y});
    uint8_t *UVPlane = reinterpret_cast<uint8_t*>(Inputs::resultY)+Inputs::resolution.y*Inputs::pitch;
    Inputs::resultUV[linear] = yuv.y;
    Inputs::resultUV[linear+1] = yuv.z;
}

/*
class Range
{
    private:
    uint2 range{UCHAR_MAX, 0};
    
    public:
    __device__ void add(uint value)
    {
        range.x = umin(value, range.x);
        range.y = umax(value, range.y);
    }   

    __device__ uint8_t distance()
    {
        return range.y-range.x;
    } 
};
*/

class Range
{
    private:   
    float m{0};
    float m2{0};
    int count{0}; 

    public:
    __device__ void add(uint value)
    { 
        m2 += value*value;
        m += value;
        count++;
    }   

    __device__ float distance()   
    {
        return 1.f/(count-1)*( m2 - (1.f/count)*m*m);
    } 
};

__device__ int2 focusCoords(int viewID, int2 coords, float focus)
{
    float2 offset = Inputs::offsets[viewID];
    float2 newCoords{__fmaf_rn(offset.x, focus, coords.x), __fmaf_rn(offset.y, focus, coords.y)};
    return {static_cast<int>(lroundf(newCoords.x)), static_cast<int>(lroundf(newCoords.y))};
}

__device__ int2 clampCoords(int2 coords)
{
    int2 result;
    result.x = min(Inputs::resolution.x-1, max(1, coords.x));
    result.y = min(Inputs::resolution.y-1, max(1, coords.y));
    return result;
}

__global__ void perPixelKernel(uint8_t *result)
{
    int2 coords = getImgCoords();
    if(coordsOutside(coords, Inputs::resolution))
        return;
    int l = linearCoords(coords, {Inputs::resolution});
 
    float bestFocus{0}; 
    float bestDispersion{99999999.0f}; 
    float focus = Inputs::focusRange.x;
    for(int f=0; f<Inputs::FOCUS_STEPS; f++)
    {
        Range range[KERNEL_WIDTH][KERNEL_WIDTH];
        for(int i=0; i<INPUT_COUNT; i++)
        {
            int2 focusedCoords = focusCoords(i, coords, focus);
            focusedCoords.y -= KERNEL;
            focusedCoords = clampCoords(focusedCoords);
            for(int k=0; k<KERNEL_WIDTH; k++)
            {   
                //TODO clamp maybe here to avoid shift at edges - test
                focusedCoords.y++;
                focusedCoords.y = min(Inputs::resolution.y, focusedCoords.y);
                uint3 sample = loadClosestY(i, focusedCoords);
                uint8_t *pixels = reinterpret_cast<uint8_t*>(&sample)+sample.z; 
                for(int p=0; p<KERNEL_WIDTH; p++)
                    range[k][p].add(pixels[p]);
            }
        }

        float dispersion{0};
        for(int i=0; i<KERNEL_WIDTH; i++)
            for(int j=0; j<KERNEL_WIDTH; j++)
                dispersion += range[i][j].distance(); 
        if(dispersion < bestDispersion)
        {
            bestDispersion = dispersion;
            bestFocus = focus;
        } 
        focus += Inputs::focusStep;
    }
    
    /*    
    if(coords.x == coords.y && coords.x == 0)
        printf("%f %f \n", Inputs::focusRange.x, Inputs::focusRange.y);
    static int ff=0;
    float f=Inputs::focusRange.x+ff*0.0000001;
    ff++;
    if(coords.x == coords.y && coords.x == 0)
        printf("%f \n", f);
    bestFocus = f;
    */
        

    float3 yuv{0,0,0};
    for(int i=0; i<INPUT_COUNT; i++)
    {
        int2 focusedCoords = focusCoords(i, coords, bestFocus);
        focusedCoords = clampCoords(focusedCoords);
        yuv.x = __fmaf_rn(  loadY(i, focusedCoords),
                            Inputs::weights[i], yuv.x);
        uchar2 uv = loadUV(i, focusedCoords); 
        yuv.y = __fmaf_rn(  uv.x,
                            Inputs::weights[i], yuv.y);
        yuv.z = __fmaf_rn(  uv.y,
                            Inputs::weights[i], yuv.z);
    }
    yuv.x *= Inputs::inverseWeightSum;
    yuv.y *= Inputs::inverseWeightSum;
    yuv.z *= Inputs::inverseWeightSum;
    //yuv.x=((bestFocus-Inputs::focusRange.x)/(Inputs::focusRange.y-Inputs::focusRange.x))*255;
    //yuv.y = yuv.z = 128;
    store({static_cast<uint8_t>(round(yuv.x)), static_cast<uint8_t>(round(yuv.y)), static_cast<uint8_t>(round(yuv.z))}, coords);
}

void perPixel(std::vector<CUdeviceptr> inFrames, std::vector<float> inWeights, std::vector<float2> inOffsets, std::vector<int> inPitches, uint8_t *result, float weightSum, int2 resolution, int pitch, float2 focusRange)
{
    cudaMemcpyToSymbol(Inputs::weights, inWeights.data(), INPUT_COUNT * sizeof(float));
    cudaMemcpyToSymbol(Inputs::pitches, inPitches.data(), INPUT_COUNT * sizeof(int));
    cudaMemcpyToSymbol(Inputs::offsets, inOffsets.data(), INPUT_COUNT * sizeof(float2));
    cudaMemcpyToSymbol(Inputs::focusRange, &focusRange, sizeof(int2));
    float focusStep{(focusRange.y-focusRange.x)/Inputs::FOCUS_STEPS};
    cudaMemcpyToSymbol(Inputs::focusStep, &focusStep, sizeof(int));
    cudaMemcpyToSymbol(Inputs::inverseWeightSum, &weightSum, sizeof(float));
    cudaMemcpyToSymbol(Inputs::resolution, &resolution, sizeof(int2));
    cudaMemcpyToSymbol(Inputs::pitch, &pitch, sizeof(int));
    int2 doubleResolution{resolution.x*2, resolution.y*2};
    cudaMemcpyToSymbol(Inputs::doubleResolution, &doubleResolution, sizeof(int2));
    int pixelCount{pitch * resolution.y};
    cudaMemcpyToSymbol(Inputs::pixelCount, &pixelCount, sizeof(int));
    std::vector<int> pixelCounts; 
    for(auto const &p : inPitches)
        pixelCounts.push_back(p * resolution.y);
    cudaMemcpyToSymbol(Inputs::pixelCounts, pixelCounts.data(), INPUT_COUNT * sizeof(int));
    cudaMemcpyToSymbol(Inputs::framesY, inFrames.data(), INPUT_COUNT * sizeof(CUdeviceptr));
    std::vector<size_t> UVFrames;
    for(int i=0; i<INPUT_COUNT; i++)
    {
        UVFrames.push_back(inFrames[i]+pixelCounts[i]);
    }
    cudaMemcpyToSymbol(Inputs::framesUV, inFrames.data(), INPUT_COUNT * sizeof(CUdeviceptr));
    cudaMemcpyToSymbol(Inputs::resultY, &result, sizeof(uint8_t*));
    uint8_t* UVResult = result+pixelCount;
    cudaMemcpyToSymbol(Inputs::resultUV, &UVResult, sizeof(uint8_t*));
    constexpr dim3 WG_SIZE{16, 16, 1};
    dim3 wgCount{1 + resolution.x / WG_SIZE.x, 1 + resolution.y / WG_SIZE.y, 1};
    perPixelKernel<<<wgCount, WG_SIZE, 0>>>(result);
}

}

namespace Conversion
{
constexpr bool FLOAT_YUV_CONVERSION{false};
class NV12Block
{
    public:
        static constexpr int BLOCK_SIZE{4};
        //source: https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering
        static constexpr float4 yuvCoefsFloat{1.596027, -0.391762, -0.812968, 2.017232};
        static constexpr float yuvCommonCoefFloat{1.164383};
        static constexpr int4 yuvCoefsInt{409, -100, -208, 516};
        static constexpr int yuvCommonCoefInt{298};
        int linearCoordsY[4];
        int linearCoordsUV;
        int Y[BLOCK_SIZE];
        int U;
        int V;

        __device__ uchar4 RGBToRGBA8(int3 RGB)
        {
            return{ static_cast<unsigned char>(RGB.x),
                    static_cast<unsigned char>(RGB.y),
                    static_cast<unsigned char>(RGB.z),
                    255};
        }

        __device__ void initCoords(int2 coords, int2 doubleCoords, int2 resolution, int2 halfResolution, int pitch)
        {
            int linear = linearCoords(doubleCoords, {pitch, resolution.y});
            linearCoordsY[0] = linear;
            linearCoordsY[1] = linear + 1;
            linear = linearCoords({doubleCoords.x, doubleCoords.y + 1}, {pitch, resolution.y});
            linearCoordsY[2] = linear;
            linearCoordsY[3] = linear + 1;
            linearCoordsUV = linearCoords({doubleCoords.x, coords.y}, {pitch, resolution.y});
        }

        __device__ void load(uint8_t *NV12, int pixelCount)
        {
            for(int i = 0; i < BLOCK_SIZE; i++)
                Y[i] = NV12[linearCoordsY[i]];
            auto UVplane = NV12 + pixelCount;
            U = UVplane[linearCoordsUV];
            V = UVplane[linearCoordsUV + 1];
        }

        __device__ int clamp8Bit(float value)
        {
            if(value > 255)
                return 255;
            if(value < 0)
                return 0;
            return value;
        }

        __device__ int3 YRuvGuvBuvToRGB(float Y, float Ruv, float Guv, float Buv)
        {
            int R = clamp8Bit(round(Y + Ruv));
            int G = clamp8Bit(round(Y + Guv));
            int B = clamp8Bit(round(Y + Buv));
            return {R, G, B};
        }

        __device__ int3 YRuvGuvBuvToRGB(int Y, int Ruv, int Guv, int Buv)
        {
            int R = clamp8Bit((Y + Ruv + 128) >> 8);
            int G = clamp8Bit((Y + Guv + 128) >> 8);
            int B = clamp8Bit((Y + Buv + 128) >> 8);
            return {R, G, B};
        }

        template <typename T>
        __device__ void storeRGBA(cudaSurfaceObject_t RGBA, int2 blockCoords)
        {
            int UN = U - 128;
            int VN = V - 128;
            T ruv, buv, guv, coef;
            if constexpr(std::is_same<T, float>::value)
            {
                ruv = VN * yuvCoefsFloat.x;
                buv = UN * yuvCoefsFloat.y + VN * yuvCoefsFloat.z;
                guv = UN * yuvCoefsFloat.w;
                coef = yuvCommonCoefFloat;
            }
            else if constexpr(std::is_same<T, int>::value)
            {
                ruv = VN * yuvCoefsInt.x;
                buv = UN * yuvCoefsInt.y + VN * yuvCoefsInt.z;
                guv = UN * yuvCoefsInt.w;
                coef = yuvCommonCoefInt;
            }

            const int2 offsets[4] { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
            for(int i = 0; i < BLOCK_SIZE; i++)
            {
                uchar4 data = RGBToRGBA8(YRuvGuvBuvToRGB((Y[i] - 16) * coef, ruv, buv, guv));
                surf2Dwrite(data, RGBA, (blockCoords.x + offsets[i].x)*sizeof(uchar4), blockCoords.y - offsets[i].y - 1, cudaBoundaryModeClamp);
            }
        }
};

__global__ void NV12ToRGBAKernel(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int2 halfResolution, int pixelCount, int pitch)
{
    int2 coords = getImgCoords();
    if(coordsOutside(coords, halfResolution))
        return;
    int2 doubleCoords{coords.x * 2, coords.y * 2};

    NV12Block block;
    block.initCoords(coords, doubleCoords, resolution, halfResolution, pitch);
    block.load(NV12, pixelCount);
    if constexpr(FLOAT_YUV_CONVERSION)
        block.storeRGBA<float>(RGBA, {doubleCoords.x, resolution.y - doubleCoords.y});
    else
        block.storeRGBA<int>(RGBA, {doubleCoords.x, resolution.y - doubleCoords.y});
}

void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int pitch)
{
    int2 halfResolution{resolution.x >> 1, resolution.y >> 1};
    constexpr dim3 WG_SIZE{16, 16, 1};
    dim3 wgCount{1 + halfResolution.x / WG_SIZE.x, 1 + halfResolution.y / WG_SIZE.y, 1};
    NV12ToRGBAKernel<<<wgCount, WG_SIZE, 0>>>(NV12, RGBA, resolution, halfResolution, pitch * resolution.y, pitch);
}
}


