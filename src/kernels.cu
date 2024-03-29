#include <stdint.h>
#include <stdio.h>
#include "kernels.h"

[[nodiscard]] __device__ int linearCoords(int2 coords, int2 resolution)
{
    int linearCoords = coords.y * resolution.x + coords.x;
    return linearCoords;
}
[[nodiscard]] __device__ bool coordsOutside(int2 coords, int2 resolution)
{
    return (coords.x >= resolution.x || coords.y >= resolution.y);
}

[[nodiscard]] __device__ int2 getImgCoords()
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
__constant__ uint8_t *resultY;
__constant__ uint8_t *resultUV;
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
constexpr int KERNEL_WIDTH{KERNEL * 2 + 1};

[[nodiscard]] __device__ uint8_t loadY(int frameID, int2 coords)
{
    int linear = linearCoords(coords, {Inputs::pitches[frameID], Inputs::resolution.y});
    return reinterpret_cast<uint8_t *>(Inputs::framesY[frameID])[linear];
}

[[nodiscard]] __device__ uint3 loadClosestY(int frameID, int2 coords)
{
    constexpr int UNIT_SIZE{4};
    int linear = linearCoords(coords, {Inputs::pitches[frameID], Inputs::resolution.y}) - KERNEL;
    if(linear < 0)
        linear = 0;
    int roundLinear = linear / UNIT_SIZE;
    uint offset = linear - roundLinear * UNIT_SIZE;
    uint2 sample;
    sample.x = reinterpret_cast<uint *>(Inputs::framesY[frameID])[roundLinear];
    sample.y = reinterpret_cast<uint *>(Inputs::framesY[frameID])[roundLinear + 1];
    return {sample.x, sample.y, offset};
}

[[nodiscard]] __device__ uchar2 loadUV(int frameID, int2 coords)
{
    int linear = linearCoords({coords.x - (coords.x & 1), coords.y >> 1}, {Inputs::pitches[frameID], Inputs::resolution.y});
    uint8_t *UVPlane = reinterpret_cast<uint8_t *>(Inputs::framesY[frameID]) + Inputs::pixelCounts[frameID];
    return {UVPlane[linear], UVPlane[linear + 1]};
}

__device__ void store(uchar3 yuv, int2 coords)
{
    int linear = linearCoords(coords, {Inputs::pitch, Inputs::resolution.y});
    Inputs::resultY[linear] = yuv.x;
    linear = linearCoords({coords.x - (coords.x & 1), coords.y >> 1}, {Inputs::pitch, Inputs::resolution.y});
    Inputs::resultUV[linear] = yuv.y;
    Inputs::resultUV[linear + 1] = yuv.z;
}

/*
//VAR
class Dispersion
{
    private:
    float m{0};
    float m2{0};
    static constexpr int COUNT{4};

    public:
    __device__ void add(uint value)
    {
        m2 += value*value;
        m += value;
    }

    __device__ float distance()
    {
        return 1.f/(COUNT-1)*( m2 - (1.f/COUNT)*m*m);
    }
};
*/

/*
//RANGE
class Dispersion
{
    private:
    uchar2 minMax{255,0};

    public:
    __device__ void add(uint value)
    {
        if(value > minMax.y)
            minMax.y = value;
        else if(value < minMax.x)
            minMax.x = value;
    }

    __device__ float distance()
    {
        return minMax.y-minMax.x;
    }
};
*/

//MAD
class Dispersion
{
    private:
        static constexpr int COUNT{4};
        int values[COUNT];
        int id{0};

    public:
        __device__ void add(int value)
        {
            values[id] = value;
            id++;
        }

        [[nodiscard]] __device__ float distance() const
        {
            int result{0};
            //result += abs(values[0]-values[1]);
            result = __sad(values[0], values[1], result);
            result = __sad(values[0], values[2], result);
            result = __sad(values[0], values[3], result);
            result = __sad(values[1], values[2], result);
            result = __sad(values[1], values[3], result);
            result = __sad(values[2], values[3], result);
            return result;
        }
};

class BlockDispersion
{
    private:
        static constexpr int COUNT{4};
        static constexpr int COLOR_WEIGHT{KERNEL_WIDTH * KERNEL_WIDTH};
        uint8_t values[KERNEL_WIDTH][KERNEL_WIDTH][COUNT];
        uchar2 colors[COUNT];

    public:
        __device__ void addColor(int position, uchar2 value)
        {
            colors[position] = value;
        }

        __device__ void add(int3 position, uint8_t value)
        {
            values[position.x][position.y][position.z] = value;
        }

        [[nodiscard]] __device__ float colorDispersion() const
        {
            Dispersion colorRange[2];
            for(int k = 0; k < COUNT; k++)
            {
                colorRange[0].add(colors[k].x);
                colorRange[1].add(colors[k].y);
            }
            float dispersion{0};
            dispersion = colorRange[0].distance() + colorRange[1].distance();
            return dispersion;
        }

        [[nodiscard]] __device__ float dispersionOverElements() const
        {
            float dispersion{0};
            for(int i = 0; i < KERNEL_WIDTH; i++)
                for(int j = 0; j < KERNEL_WIDTH; j++)
                {
                    const uint8_t *vals = values[i][j];
                    Dispersion range;
                    for(int k = 0; k < COUNT; k++)
                        range.add(vals[k]);
                    dispersion += range.distance();
                }
            return dispersion;
        }

        [[nodiscard]] __device__ float interElementMultiplier() const
        {
            int tests[COUNT] {0, 0, 0, 0};
            for(int k = 0; k < COUNT; k++)
            {
                tests[k] |= (uint32_t)(values[0][0][k] >= values[4][0][k])<<0u;
                tests[k] |= (uint32_t)(values[0][0][k] >= values[0][4][k])<<1u;
                tests[k] |= (uint32_t)(values[4][0][k] >= values[4][4][k])<<2u;
                tests[k] |= (uint32_t)(values[0][4][k] >= values[4][4][k])<<3u;
                                                                         
                tests[k] |= (uint32_t)(values[0][0][k] >= values[4][4][k])<<4u;
                tests[k] |= (uint32_t)(values[4][0][k] >= values[0][4][k])<<5u;
                                                                         
                tests[k] |= (uint32_t)(values[2][1][k] >= values[2][3][k])<<6u;
                tests[k] |= (uint32_t)(values[1][2][k] >= values[3][2][k])<<7u;
                                                                         
                tests[k] |= (uint32_t)(values[0][1][k] >= values[4][3][k])<<8u;
                tests[k] |= (uint32_t)(values[0][3][k] >= values[4][1][k])<<9u;
                tests[k] |= (uint32_t)(values[1][0][k] >= values[3][4][k])<<10u;
                tests[k] |= (uint32_t)(values[3][0][k] >= values[1][4][k])<<11u;
                                                                         
                tests[k] |= (uint32_t)(values[0][2][k] >= values[4][2][k])<<12u;
                tests[k] |= (uint32_t)(values[2][0][k] >= values[2][4][k])<<13u;
                                                                         
                tests[k] |= (uint32_t)(values[1][1][k] >= values[1][3][k])<<14u;
                tests[k] |= (uint32_t)(values[1][3][k] >= values[3][3][k])<<15u;
                tests[k] |= (uint32_t)(values[3][3][k] >= values[3][1][k])<<16u;
                tests[k] |= (uint32_t)(values[3][1][k] >= values[1][1][k])<<17u;
                                                                         
                tests[k] |= (uint32_t)(values[1][1][k] >= values[3][3][k])<<18u;
                tests[k] |= (uint32_t)(values[3][1][k] >= values[1][3][k])<<19u;

            }
            constexpr float TEST_COUNT_INV{1.0f / 20};
            unsigned int allTests[2] = {~0U, ~0U};
            for(int k = 0; k < COUNT; k++)
            {
                allTests[0] &= tests[k];
                allTests[1] &= ~tests[k];
            }
            int count = __popc((allTests[0] | allTests[1]) & 0x000fffffu);
            return (1.0f - count * TEST_COUNT_INV);
        }

        [[nodiscard]] __device__ float dispersion() const
        {
            float dispersion{0};
            dispersion = dispersionOverElements();
            dispersion += colorDispersion() * COLOR_WEIGHT;
            return dispersion * interElementMultiplier();
        }
};

[[nodiscard]] __device__ int2 focusCoords(int viewID, int2 coords, float focus)
{
    float2 offset = Inputs::offsets[viewID];
    float2 newCoords{__fmaf_rn(offset.x, focus, coords.x), __fmaf_rn(offset.y, focus, coords.y)};
    return {static_cast<int>(lroundf(newCoords.x)), static_cast<int>(lroundf(newCoords.y))};
}

[[nodiscard]] __device__ int2 clampCoords(int2 coords)
{
    int2 result;
    //result.x = min(Inputs::resolution.x-1, max(0, coords.x));
    //result.y = min(Inputs::resolution.y-1, max(0, coords.y));
    result.x = __vimin_s32_relu(Inputs::resolutionMinusOne.x, coords.x);
    result.y = __vimin_s32_relu(Inputs::resolutionMinusOne.y, coords.y);
    return result;
}

[[nodiscard]] __device__ float optimalFocus(int2 coords)
{
    float bestFocus{0};
    float bestDispersion{9999999.0f};
    float focus = Inputs::focusRange.x;
    for(int f = 0; f < Inputs::FOCUS_STEPS; f++)
    {
        BlockDispersion block;
        for(int i = 0; i < INPUT_COUNT; i++)
        {
            int2 focusedCoords = focusCoords(i, coords, focus);
            focusedCoords.y -= KERNEL;
            focusedCoords = clampCoords(focusedCoords);
            uchar2 color = loadUV(i, focusedCoords);
            block.addColor(i, color);
            for(int k = 0; k < KERNEL_WIDTH; k++)
            {
                focusedCoords.y++;
                focusedCoords.y = min(Inputs::resolutionMinusOne.y, focusedCoords.y);
                uint3 sample = loadClosestY(i, focusedCoords);
                uint8_t *pixels = reinterpret_cast<uint8_t *>(&sample) + sample.z;
                for(int p = 0; p < KERNEL_WIDTH; p++)
                    block.add({k, p, i}, pixels[p]);
            }
        }

        float dispersion = block.dispersion();
        if(dispersion < bestDispersion)
        {
            bestDispersion = dispersion;
            bestFocus = focus;
        }
        focus += Inputs::focusStep;
    }
    return bestFocus;
}

[[nodiscard]] __device__ uchar3 focusedColor(int2 coords, float focus)
{
    float3 yuv{0, 0, 0};
    for(int i = 0; i < INPUT_COUNT; i++)
    {
        int2 focusedCoords = focusCoords(i, coords, focus);
        focusedCoords = clampCoords(focusedCoords);
        yuv.x = __fmaf_rn(loadY(i, focusedCoords),
                          Inputs::weights[i], yuv.x);
        uchar2 uv = loadUV(i, focusedCoords);
        yuv.y = __fmaf_rn(uv.x,
                          Inputs::weights[i], yuv.y);
        yuv.z = __fmaf_rn(uv.y,
                          Inputs::weights[i], yuv.z);
    }
    yuv.x *= Inputs::inverseWeightSum;
    yuv.y *= Inputs::inverseWeightSum;
    yuv.z *= Inputs::inverseWeightSum;
    return {static_cast<uint8_t>(round(yuv.x)), static_cast<uint8_t>(round(yuv.y)), static_cast<uint8_t>(round(yuv.z))};
}

__global__ void perPixelKernel(uint8_t *result)
{
    int2 coords = getImgCoords();
    if(coordsOutside(coords, Inputs::resolution))
        return;
    float bestFocus = optimalFocus(coords);
    uchar3 color = focusedColor(coords, bestFocus);

    //yuv.x=((bestFocus-Inputs::focusRange.x)/(Inputs::focusRange.y-Inputs::focusRange.x))*255;
    //yuv.y = yuv.z = 128;
    store(color, coords);
}

void perPixel(std::vector<CUdeviceptr> inFrames, std::vector<float> inWeights, std::vector<float2> inOffsets, std::vector<int> inPitches, uint8_t *result, float weightSum, int2 resolution, int pitch, float2 focusRange)
{
    cudaMemcpyToSymbol(Inputs::weights, inWeights.data(), INPUT_COUNT * sizeof(float));
    cudaMemcpyToSymbol(Inputs::pitches, inPitches.data(), INPUT_COUNT * sizeof(int));
    cudaMemcpyToSymbol(Inputs::offsets, inOffsets.data(), INPUT_COUNT * sizeof(float2));
    cudaMemcpyToSymbol(Inputs::focusRange, &focusRange, sizeof(int2));
    float focusStep{(focusRange.y - focusRange.x) / Inputs::FOCUS_STEPS};
    cudaMemcpyToSymbol(Inputs::focusStep, &focusStep, sizeof(int));
    cudaMemcpyToSymbol(Inputs::inverseWeightSum, &weightSum, sizeof(float));
    cudaMemcpyToSymbol(Inputs::resolution, &resolution, sizeof(int2));
    int2 resolutionMinusOne{resolution.x - 1, resolution.y - 1};
    cudaMemcpyToSymbol(Inputs::resolutionMinusOne, &resolutionMinusOne, sizeof(int2));
    cudaMemcpyToSymbol(Inputs::pitch, &pitch, sizeof(int));
    int2 doubleResolution{resolution.x * 2, resolution.y * 2};
    cudaMemcpyToSymbol(Inputs::doubleResolution, &doubleResolution, sizeof(int2));
    int pixelCount{pitch * resolution.y};
    cudaMemcpyToSymbol(Inputs::pixelCount, &pixelCount, sizeof(int));
    std::vector<int> pixelCounts;
    for(auto const &p : inPitches)
        pixelCounts.push_back(p * resolution.y);
    cudaMemcpyToSymbol(Inputs::pixelCounts, pixelCounts.data(), INPUT_COUNT * sizeof(int));
    cudaMemcpyToSymbol(Inputs::framesY, inFrames.data(), INPUT_COUNT * sizeof(CUdeviceptr));
    std::vector<size_t> UVFrames;
    for(int i = 0; i < INPUT_COUNT; i++)
    {
        UVFrames.push_back(inFrames[i] + pixelCounts[i]);
    }
    cudaMemcpyToSymbol(Inputs::framesUV, inFrames.data(), INPUT_COUNT * sizeof(CUdeviceptr));
    cudaMemcpyToSymbol(Inputs::resultY, &result, sizeof(uint8_t *));
    uint8_t *UVResult = result + pixelCount;
    cudaMemcpyToSymbol(Inputs::resultUV, &UVResult, sizeof(uint8_t *));
    constexpr dim3 WG_SIZE{16, 16, 1};
    dim3 wgCount{1 + resolution.x / WG_SIZE.x, 1 + resolution.y / WG_SIZE.y, 1};
    perPixelKernel <<< wgCount, WG_SIZE, 0>>>(result);
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

        [[nodiscard]] __device__ uchar4 RGBToRGBA8(int3 RGB) const
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

        [[nodiscard]] __device__ int clamp8Bit(float value) const
        {
            if(value > 255)
                return 255;
            if(value < 0)
                return 0;
            return value;
        }

        [[nodiscard]] __device__ int3 YRuvGuvBuvToRGB(float Y, float Ruv, float Guv, float Buv) const
        {
            int R = clamp8Bit(round(Y + Ruv));
            int G = clamp8Bit(round(Y + Guv));
            int B = clamp8Bit(round(Y + Buv));
            return {R, G, B};
        }

        [[nodiscard]] __device__ int3 YRuvGuvBuvToRGB(int Y, int Ruv, int Guv, int Buv) const
        {
            int R = clamp8Bit((Y + Ruv + 128) >> 8);
            int G = clamp8Bit((Y + Guv + 128) >> 8);
            int B = clamp8Bit((Y + Buv + 128) >> 8);
            return {R, G, B};
        }

        template <typename T>
        __device__ void storeRGBA(cudaSurfaceObject_t RGBA, int2 blockCoords) const
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
    NV12ToRGBAKernel <<< wgCount, WG_SIZE, 0>>>(NV12, RGBA, resolution, halfResolution, pitch * resolution.y, pitch);
}
}
