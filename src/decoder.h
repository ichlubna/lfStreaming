#include <string>
#include <vector>
#include <memory>
#include "renderer.h"
#include "videoDecoder.h"
#include "interpolator.h"
#include "cudaGLInterop.h"

class Decoder
{
    public:
    Decoder(std::string input, size_t startFrame=0);
    void decodeAndPlay();
    void decodeAndStore(std::string trajectory);

    private:
    class SelectedFrames
    {
        public:
        class InterpolationInfo
        {
            public:
            static constexpr size_t FRAME_COUNT{4};
            static constexpr size_t WEIGHTS_COUNT{3};
            glm::ivec2 positions[FRAME_COUNT];
            int idOrder[FRAME_COUNT];
            float weights[WEIGHTS_COUNT];
        }; 

        class Frame
        {
            public:
            glm::ivec2 coord;
            float weight;
        };

        enum Order {TOP_BOTTOM=0, LEFT_RIGHT=1, DIAGONAL=2};
        enum FramePosition {TOP_LEFT=0, TOP_RIGHT=1, BOTTOM_LEFT=2, BOTTOM_RIGHT=3};
        Frame frames[InterpolationInfo::FRAME_COUNT];

        void compute(glm::uvec2 gridSize, glm::vec2 position);
        float interWeight(Order first, Order second) const {return frames[second].weight/(frames[first].weight+frames[second].weight);}
        InterpolationInfo guide(Order order); 

        //glm::vec3 topBottomMiddleInter() const { float top = topInter(); float bottom = bottomInter(); return {top, bottom, bottom/(top+bottom)};}
        //glm::vec3 leftRightMiddleInter() const { float left = leftInter(); float right = rightInter(); return {left, right, right/(left+right)};}
    };

    const int orderIDs[3][4] { {0,1,2,3}, {0,2,1,3}, {0,3,1,2} };
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<VideoDecoder> videoDecoder;
    std::unique_ptr<Interpolator> interpolator;
    std::unique_ptr<CudaGLInterop> interop;
    std::vector<glm::vec2> parseTrajectory(std::string textTrajectory) const;
    class IntermediateFrame 
    {
        public:
        IntermediateFrame(glm::ivec2 resolution);
        CUdeviceptr frame;
        size_t pitch{0};
        void* ptr() {return reinterpret_cast<void*>(&frame);}
    };
    std::vector<IntermediateFrame> intermediateFrames;
    SelectedFrames pickFromGrid(glm::uvec2 gridSize, glm::vec2 position) const;
    CUdeviceptr decodeAndInterpolate(glm::vec2 position);
    void prepareFrames();
    std::vector<void*> getIntermediatePtrs();
};
