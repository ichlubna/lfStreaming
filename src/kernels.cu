#include <stdint.h>

namespace Conversion
{
    constexpr bool FLOAT_YUV_CONVERSION{false};

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

    __device__ int linearCoords(int2 coords, int2 resolution)
    {
        int linearCoords = coords.y*resolution.x + coords.x;
        return linearCoords; 
    }

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
            int linear = linearCoords(doubleCoords, {pitch,resolution.y});
            linearCoordsY[0] = linear;
            linearCoordsY[1] = linear+1;
            linear = linearCoords({doubleCoords.x, doubleCoords.y+1}, {pitch,resolution.y});
            linearCoordsY[2] = linear;
            linearCoordsY[3] = linear+1;
            linearCoordsUV = linearCoords({doubleCoords.x, coords.y}, {pitch,resolution.y});
        }

        __device__ void load(uint8_t *NV12, int pixelCount)
        { 
            for(int i=0; i<BLOCK_SIZE; i++)
                Y[i] = NV12[linearCoordsY[i]]; 
            auto UVplane = NV12+pixelCount;
            U = UVplane[linearCoordsUV];
            V = UVplane[linearCoordsUV+1];
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
            int UN = U-128;
            int VN = V-128;
            T ruv, buv, guv, coef;
            if constexpr (std::is_same<T, float>::value)
            {
                ruv = VN*yuvCoefsFloat.x; 
                buv = UN*yuvCoefsFloat.y + VN*yuvCoefsFloat.z; 
                guv = UN*yuvCoefsFloat.w;
                coef = yuvCommonCoefFloat;
            }
            else if constexpr (std::is_same<T, int>::value)
            {
                ruv = VN*yuvCoefsInt.x;
                buv = UN*yuvCoefsInt.y + VN*yuvCoefsInt.z;
                guv = UN*yuvCoefsInt.w;
                coef = yuvCommonCoefInt;
            }

            const int2 offsets[4]{ {0,0}, {0,1}, {1,0}, {1,1} };
            for(int i=0; i<BLOCK_SIZE; i++)
            {
                uchar4 data = RGBToRGBA8(YRuvGuvBuvToRGB((Y[i]-16)*coef, ruv, buv, guv));
                surf2Dwrite(data, RGBA, (blockCoords.x+offsets[i].x)*sizeof(uchar4), blockCoords.y+offsets[i].y, cudaBoundaryModeClamp);
            }
        }
    };

    __global__ void NV12ToRGBAKernel(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int2 halfResolution, int pixelCount, int pitch)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords, halfResolution))
            return;
        int2 doubleCoords{coords.x*2, coords.y*2};

        NV12Block block;
        block.initCoords(coords, doubleCoords, resolution, halfResolution, pitch);
        block.load(NV12, pixelCount);
        if constexpr (FLOAT_YUV_CONVERSION)
            block.storeRGBA<float>(RGBA, {doubleCoords.x, resolution.y-1-doubleCoords.y}); 
        else
            block.storeRGBA<int>(RGBA, {doubleCoords.x, resolution.y-1-doubleCoords.y}); 
    }
    
    void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int pitch)
    {
        constexpr dim3 WG_SIZE{16,16,1};
        dim3 wgCount{1+resolution.x/WG_SIZE.x/2, 1+resolution.y/WG_SIZE.y/2, 1};
        NV12ToRGBAKernel<<<wgCount, WG_SIZE, 0>>>(NV12, RGBA, resolution, {resolution.x/2, resolution.y/2}, pitch*resolution.y, pitch);
    }
}
