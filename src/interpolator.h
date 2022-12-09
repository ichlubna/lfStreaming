#include <vector>
#include <glm/glm.hpp>
#include "libs/fruc/NvOFFRUC.h"

class Interpolator
{
    public:
        Interpolator(glm::ivec2 inputResolution);
        ~Interpolator();
        void init();
        class Pair
        {
            public:
                void *first;
                void *second;
                glm::uvec2 pitch;
                float weight;
                void *output;
        };
        void interpolate(std::vector<Pair> data);
        void registerResources(std::vector<void *> *resources);
        void unregisterResources(std::vector<void *> *resources);

    private:
        enum Os {LINUX};
        static constexpr Os OS{LINUX};
        static constexpr char NvOFFRCULibraryPath[] {"../src/libs/fruc/libNvOFFRUC.so"};
        glm::ivec2 resolution;
        size_t frameNumber{0};
        NvOFFRUC_REGISTER_RESOURCE_PARAM registerParam{};

        void *getProc(void *handle, const char *symbol);
        void loadLibrary();
        void initInterpolation();
        void process(float factor, bool updateOnly, void *input, void *output, size_t pitch);

        NvOFFRUCHandle fruc;
        PtrToFuncNvOFFRUCCreate NvOFFRUCCreate{nullptr};
        PtrToFuncNvOFFRUCRegisterResource NvOFFRUCRegisterResource{nullptr};
        PtrToFuncNvOFFRUCUnregisterResource NvOFFRUCUnregisterResource{nullptr};
        PtrToFuncNvOFFRUCProcess NvOFFRUCProcess{nullptr};
        PtrToFuncNvOFFRUCDestroy NvOFFRUCDestroy{nullptr};

};
