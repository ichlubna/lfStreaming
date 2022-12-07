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
    enum Order {TOP_BOTTOM=0, LEFT_RIGHT, DIAGONAL};
    void setOrder(Order o) {order=o;}

    private:
    class SelectedFrames
    {
        public:
        class Frame
        {
            public:
            glm::ivec2 coord;
            float weight;
        };
        Frame topLeft, topRight, bottomLeft, bottomRight;
        float topInter() const {return topRight.weight/(topLeft.weight+topRight.weight);}
        float bottomInter() const {return bottomRight.weight/(bottomLeft.weight+bottomRight.weight);} 
        float leftInter() const {return bottomLeft.weight/(topLeft.weight+bottomLeft.weight);}
        float rightInter() const {return bottomRight.weight/(topRight.weight+bottomRight.weight);} 
        glm::vec3 topBottomMiddleInter() const { float top = topInter(); float bottom = bottomInter(); return {top, bottom, bottom/(top+bottom)};}
        glm::vec3 leftRightMiddleInter() const { float left = leftInter(); float right = rightInter(); return {left, right, right/(left+right)};}
    };

    const int orderIDs[3][4] { {0,1,2,3}, {0,2,1,3}, {0,3,1,2} };
    Order order{TOP_BOTTOM};
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
        void* ptr() {return reinterpret_cast<void*>(&frame);}
    };
    std::vector<IntermediateFrame> intermediateFrames;
    SelectedFrames pickFromGrid(glm::uvec2 gridSize, glm::vec2 position) const;
    void interpolateView(glm::vec2 position);
    void prepareFrames();
    std::vector<void*> getIntermediatePtrs();
};
