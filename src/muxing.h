#include "glm/glm.hpp"
#include <vector>
#include <set>
#include <filesystem>
#include <string>

class Muxing
{
    public:
        [[nodiscard]] static const std::set<std::filesystem::path> listPath(std::string path);
        [[nodiscard]] static glm::uvec2 parseFilename(std::string name);

        class EncodedData
        {
            public:
                EncodedData() {};
                enum Format { H265 = 0, AV1 = 1 };
                void addData(const std::vector<uint8_t> *packetData);
                void initHeader(glm::uvec2 resolution, glm::uvec2 colsRows, uint32_t format, uint32_t timeFrameCount, float aspect);
                std::vector<uint32_t> header;
                std::vector<uint8_t> packets;
                std::vector<uint32_t> offsets;
                std::vector<uint32_t> references;
                static constexpr size_t HEADER_VALUES_COUNT{7};
                static constexpr float FIXED_FLOAT_MULTIPLIER{10000.0f};
                [[nodiscard]] glm::uvec2 resolution() const
                {
                    return {header[0], header[1]};
                }
                [[nodiscard]] glm::uvec2 colsRows() const
                {
                    return {header[2], header[3]};
                }
                [[nodiscard]]  Format format() const
                {
                    return static_cast<Format>(header[4]);
                }
                [[nodiscard]] size_t timeFrameCount() const
                {
                    return header[5];
                }
                [[nodiscard]] float aspect() const
                {
                    return header[6]/FIXED_FLOAT_MULTIPLIER;
                }
                [[nodiscard]] size_t gridSize() const
                {
                    return colsRows().x * colsRows().y;
                }
        };

        class Muxer
        {
            public:
                friend void operator<<(Muxer &m, const std::vector<uint8_t> *packet)
                {
                    m.addPacket(packet);
                };
                void save(std::string filePath);
                void init(glm::uvec2 resolution, glm::uvec2 colsRows, uint32_t format, uint32_t timeFrameCount, float aspect)
                {
                    data.initHeader(resolution, colsRows, format, timeFrameCount, aspect);
                    initialized = true;
                };
                [[nodiscard]] bool isInitialized()
                {
                    return initialized;
                }
                void endTimeFrame(glm::uvec2 referenceCoords);

            private:
                [[nodiscard]] size_t getLinearIndex(glm::ivec3 colsRowsTime) const;
                void addPacket(const std::vector<uint8_t> *packetData);
                EncodedData data;
                bool initialized{false};
                size_t frameNumber{0};
        };

        class Demuxer
        {
            public:
                class PacketPointer
                {
                    public:
                        const uint8_t *data;
                        size_t size;
                };
                Demuxer(std::string filePath);
                [[nodiscard]] glm::uvec2 getResolution()
                {
                    return data.resolution();
                }
                [[nodiscard]] const PacketPointer getPacket(glm::ivec3 colsRowsTime) const;
                [[nodiscard]] const PacketPointer getReferencePacket(int time) const;
                EncodedData data;

            private:
                [[nodiscard]] std::vector<uint8_t> copyPacket(glm::ivec3 colsRowsTime) const;
                [[nodiscard]] size_t getLinearIndex(glm::ivec3 colsRowsTime) const;
        };

    private:
};

