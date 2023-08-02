#include <stdint.h>
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
__constant__ float weights[INPUT_COUNT];
__constant__ int pitches[INPUT_COUNT];
__constant__ size_t frames[INPUT_COUNT];
__constant__ float2 offsets[INPUT_COUNT];
constexpr int KERNEL{2}; 

__device__ uchar3 load(const uint8_t *NV12, int pixelCount, int2 coords, int2 resolution, int pitch)
{
    int linear = linearCoords(coords, {pitch, resolution.x});
    uchar3 yuv;
    yuv.x = NV12[linear];
    auto UVplane = NV12 + pixelCount;
    //yuv.y = UVplane[linear];
    //yuv.z = UVplane[linear + 1];
    return yuv;
}

__device__ uint8_t loadY(uint8_t *NV12, int2 coords, int2 resolution, int pitch)
{
    int linear = linearCoords(coords, {pitch, resolution.x});
    return NV12[linear];
}

__device__ uint4 loadClosestY(uint8_t *NV12, int2 coords, int2 resolution, int pitch)
{  
    constexpr int UNIT_SIZE{4}; 
    int linear = linearCoords(coords, {pitch, resolution.y}) - KERNEL;
    if(linear<0)
        linear=0;
    int roundLinear = linear/UNIT_SIZE;
    uint offset = linear-roundLinear*UNIT_SIZE;
    uint2 sample;
    sample.x = reinterpret_cast<uint*>(NV12)[roundLinear];
    sample.y = reinterpret_cast<uint*>(NV12)[roundLinear+1];
    return {sample.x, sample.y, offset, static_cast<unsigned int>(linear)};
}

__device__ uchar2 loadUV(uint8_t *NV12, int2 coords, int2 resolution, int pitch, int pixelCount)
{
    int linear = linearCoords({coords.x<<1, coords.y}, {pitch, resolution.y});
    uint8_t *UVPlane = NV12+pixelCount;
    return {UVPlane[linear], UVPlane[linear+1]}; 
}

__device__ void store(uchar3 yuv, uint8_t *target, int2 coords, int2 resolution, int pitch, int pixelCount)
{
    int linear = linearCoords(coords, {pitch, resolution.x});
    target[linear] = yuv.x;
    linear = linearCoords({coords.x<<1, coords.y}, {pitch, resolution.y});
    uint8_t *UVPlane = target+pixelCount;
    UVPlane[linear] = yuv.y;
    UVPlane[linear+1] = yuv.z;
}

class Range
{
    private:
    uchar2 range{UCHAR_MAX, 0};
    
    public:
    void add(uint8_t value)
    {
        range.x = umin(value, range.x);
        range.y = umax(value, range.y);
    }   

    uint8_t distance()
    {
        return range.y-range.x;
    } 
};

__device__ int2 focusCoords(int viewID, int coords, float focus)
{
    //enum FramePosition {TOP_LEFT = 0, TOP_RIGHT = 1, BOTTOM_LEFT = 2, BOTTOM_RIGHT = 3};
}

__global__ void perPixelKernel(uint8_t *result, int2 resolution, int pixelCount, int pitch, int2 doubleResolution)
{
    int2 coords = getImgCoords();
    if(coordsOutside(coords, resolution))
        return;
    int l = linearCoords(coords, {resolution});
    uint4 sample = loadClosestY(reinterpret_cast<uint8_t*>(frames[0]), coords, resolution, pitch);
    uint8_t *pixels = reinterpret_cast<uint8_t*>(&sample);
    uint8_t y = pixels[sample.z+KERNEL];
    uchar2 uv = loadUV(reinterpret_cast<uint8_t*>(frames[0]), coords, resolution, pitch, pixelCount); 
    store({y, uv.x, uv.y}, result, coords, resolution, pitch, pixelCount);
}

void perPixel(std::vector<CUdeviceptr> inFrames, std::vector<float> inWeights, std::vector<float2> inOffsets, std::vector<size_t> inPitches, uint8_t *result, int2 resolution, int pitch)
{
    cudaMemcpyToSymbol(PerPixelInterpolation::frames, inFrames.data(), INPUT_COUNT * sizeof(CUdeviceptr));
    cudaMemcpyToSymbol(PerPixelInterpolation::weights, inWeights.data(), INPUT_COUNT * sizeof(float));
    cudaMemcpyToSymbol(PerPixelInterpolation::pitches, inPitches.data(), INPUT_COUNT * sizeof(int));
    cudaMemcpyToSymbol(PerPixelInterpolation::offsets, inOffsets.data(), INPUT_COUNT * sizeof(float2));
    constexpr dim3 WG_SIZE{16, 16, 1};
    dim3 wgCount{1 + resolution.x / WG_SIZE.x, 1 + resolution.y / WG_SIZE.y, 1};
    perPixelKernel<<<wgCount, WG_SIZE, 0>>>(result, resolution, pitch * resolution.y, pitch, {resolution.x*2, resolution.y*2});
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


