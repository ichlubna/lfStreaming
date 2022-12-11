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
        Decoder(std::string input, size_t startFrame = 0);
        ~Decoder();
        void decodeAndPlay();
        void decodeAndStore(std::string trajectory, std::string outputPath);
        void setInterpolationMethod(std::string method);

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
                        float weights[WEIGHTS_COUNT];
                };

                class Frame
                {
                    public:
                        glm::ivec2 coord;
                        float weight;
                };

                enum Order {TOP_BOTTOM = 0, LEFT_RIGHT = 1, DIAGONAL = 2};
                enum FramePosition {TOP_LEFT = 0, TOP_RIGHT = 1, BOTTOM_LEFT = 2, BOTTOM_RIGHT = 3};
                const std::vector<FramePosition> orderIDs[3] {{TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT},
//                const std::vector<FramePosition> orderIDs[3] {{TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_RIGHT},
                    {TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT},
                    {TOP_LEFT, BOTTOM_RIGHT, TOP_RIGHT, BOTTOM_LEFT}
                };
                Frame frames[InterpolationInfo::FRAME_COUNT];

                void compute(glm::uvec2 gridSize, glm::vec2 position);
                float interWeight(Order order) const
                {
                    auto o = orderIDs[order];
                    float first = frames[o[0]].weight + frames[o[1]].weight;
                    float second = frames[o[2]].weight + frames[o[3]].weight;
                    return interWeightVal(first, second);
                }
                float interWeight(FramePosition first, FramePosition second) const
                {
                    return interWeightVal(frames[first].weight, frames[second].weight);
                }
                float interWeightVal(float first, float second) const
                {
                    return first / (first + second);
                }
                InterpolationInfo guide(Order order);
        };

        SelectedFrames framePicker;
        std::unique_ptr<Renderer> renderer;
        std::unique_ptr<VideoDecoder> videoDecoder;
        std::unique_ptr<Interpolator> interpolator;
        std::unique_ptr<CudaGLInterop> interop;
        std::vector<glm::vec2> parseTrajectory(std::string textTrajectory) const;
        std::vector<void *> intermediatePtrs;
        SelectedFrames::Order interpolationOrder{SelectedFrames::TOP_BOTTOM};
        class IntermediateFrame
        {
            public:
                IntermediateFrame(glm::ivec2 resolution);
                CUdeviceptr frame;
                size_t pitch{0};
                void *ptr()
                {
                    return reinterpret_cast<void *>(&frame);
                }
        };
        std::vector<IntermediateFrame> intermediateFrames;
        SelectedFrames pickFromGrid(glm::uvec2 gridSize, glm::vec2 position) const;
        class InterpolationResult
        {
            public:
                CUdeviceptr frame;
                size_t pitch;
        };
        template<bool measure=false>
        InterpolationResult decodeAndInterpolate(glm::vec2 position);
        void prepareFrames();
        std::vector<void *> getIntermediatePtrs();
        glm::vec2 cameraPosition();
};
