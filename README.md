# Light Field Video Streaming on GPU

This repository contains tools for light field video streaming. Visit the [research page](https://www.fit.vutbr.cz/~ichlubna/lf) for more details and the dataset. The project uses HW-accelerated NVIDIA GPU video deocders and CUDA.

Download [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk) and copy the FRUC library (libcudart.so, libNvOFFRUC.so, NvOFFRUC.h) to src/libs/fruc/.
In case of problems with loading of FRUC library create symlink in the build folder and set export path:
```
ln  libcudart.so.11.0 src/libs/fruc/libcudart.so 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
```
The project contains standard Cmake setup. Example usage:

```
cd ../scripts
chmod +x ./prepareDataset.sh
./prepareDataset.sh
mkdir build
cd ../build/
cmake ..
make -j4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
./lfEncoder -i ../scripts/finalResult/inputs/ -q 0.75 -o test.lf -a 2.276 -s 0.06_0.3
./lfStreamer -i test.lf -m PP
```


