#include <stdint.h>

namespace Conversion
{
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
        static constexpr float2 RUV{0, 1.13983};
        static constexpr float2 GUV{-0.39465, -0.58060};
        static constexpr float2 BUV{2.03211, 0};
        int linearCoordsY[4];
        int linearCoordsUV;
        float Y[BLOCK_SIZE];
        float U;
        float V;
        
        __device__ uchar4 RGBToRGBA8(float3 RGB)
        {
            return{ static_cast<unsigned char>(RGB.x),
                    static_cast<unsigned char>(RGB.y),
                    static_cast<unsigned char>(RGB.z),
                    255};
        }    
        
        __device__ void initCoords(int2 coords, int2 doubleCoords, int2 resolution, int2 halfResolution)
        { 
            int linear = linearCoords(doubleCoords, resolution);
            linearCoordsY[0] = linear;
            linearCoordsY[1] = linear+1;
            linear = linearCoords({doubleCoords.x, doubleCoords.y+1}, resolution);
            linearCoordsY[2] = linear;
            linearCoordsY[3] = linear+1;
            linearCoordsUV = linearCoords({coords.x, coords.y}, halfResolution);
        }

        __device__ void load(uint8_t *NV12, int pixelCount)
        { 
            for(int i=0; i<BLOCK_SIZE; i++)
                Y[i] = NV12[linearCoordsY[i]]; 
            auto uv = reinterpret_cast<uchar2*>(NV12+pixelCount)[linearCoordsUV];
            U = uv.x; 
            V = uv.y; 
        }

        __device__ void storeRGBA(cudaSurfaceObject_t RGBA, int2 blockCoords)
        {
            float ruv = U*RUV.x + V*RUV.y; 
            float guv = U*GUV.x + V*GUV.y; 
            float buv = U*BUV.x + V*BUV.y; 

            //auto RGBA8Ptr{reinterpret_cast<uchar4*>(RGBA)};
            const int2 offsets[4]{ {0,0}, {0,1}, {1,0}, {1,1} };
            for(int i=0; i<BLOCK_SIZE; i++)
            {
                int2 coords{blockCoords.x+offsets[i].x, blockCoords.y+offsets[i].y};
                uchar4 data = RGBToRGBA8({Y[i]+ruv, Y[i]+guv, Y[i]+buv});
                surf2Dwrite(data, RGBA, (blockCoords.x+offsets[i].x)*sizeof(uchar4), blockCoords.y+offsets[i].y);
            }
                //RGBA8Ptr[linearCoordsY[i]] = RGBToRGBA8({Y[i]+ruv, Y[i]+guv, Y[i]+buv});
        }
    };

    __global__ void NV12ToRGBAKernel(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution, int2 halfResolution, int pixelCount)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords, halfResolution))
            return;
        int2 doubleCoords{coords.x*2, coords.y*2};

        NV12Block block;
        block.initCoords(coords, doubleCoords, resolution, halfResolution);
        block.load(NV12, pixelCount);
        block.storeRGBA(RGBA, doubleCoords); 
    }
    
    void NV12ToRGBA(uint8_t *NV12, cudaSurfaceObject_t RGBA, int2 resolution)
    {
        constexpr dim3 WG_SIZE{16,16,1};
        dim3 wgCount{1+resolution.x/WG_SIZE.x/2, 1+resolution.y/WG_SIZE.y/2, 1};
        NV12ToRGBAKernel<<<wgCount, WG_SIZE, 0>>>(NV12, RGBA, resolution, {resolution.x/2, resolution.y/2}, resolution.x*resolution.y);
    }
}
