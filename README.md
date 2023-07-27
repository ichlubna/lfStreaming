# lfStreaming

Download [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk) and copy the FRUC library (libcudart.so, libNvOFFRUC.so, NvOFFRUC.h) to src/libs/fruc/.
In case of problems with loading of FRUC library create symlink in the build folder and set export path:
```
ln  libcudart.so.11.0 src/libs/fruc/libcudart.so 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
```
