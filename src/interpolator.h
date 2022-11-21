#include <glm/glm.hpp>
#include "libs/fruc/NvOFFRUC.h"

class Interpolator
{
    public:
    Interpolator(glm::ivec2 inputResolution);
    ~Interpolator();
    void init();

    private:
    enum Os {LINUX};
    static constexpr Os OS{LINUX};
    static constexpr char NvOFFRCULibraryPath[]{"../src/libs/fruc/libNvOFFRUC.so"};
    glm::ivec2 resolution;
    size_t frameNumber{0};

    void* getProc(void *handle, const char *symbol);
    void loadLibrary();
    void initInterpolation();
    void registerResources();
    void process(float factor, bool updateOnly);

    NvOFFRUCHandle fruc;
    PtrToFuncNvOFFRUCCreate NvOFFRUCCreate{nullptr};
    PtrToFuncNvOFFRUCRegisterResource NvOFFRUCRegisterResource{nullptr};
    PtrToFuncNvOFFRUCUnregisterResource NvOFFRUCUnregisterResource{nullptr};
    PtrToFuncNvOFFRUCProcess NvOFFRUCProcess{nullptr};
    PtrToFuncNvOFFRUCDestroy NvOFFRUCDestroy{nullptr};

};
