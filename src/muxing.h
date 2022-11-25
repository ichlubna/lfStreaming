#include "glm/glm.hpp"
#include <vector>
#include <set>
#include <filesystem>
#include <string>

class Muxing
{
public:
    static const std::set<std::filesystem::path> listPath(std::string path);
    static glm::uvec2 parseFilename(std::string name);

    class EncodedData
    {
    public:
        EncodedData() {};
        enum Format { H265=0, AV1=1 };
        void addData(const std::vector<uint8_t> *packetData);
        void initHeader(glm::uvec2 resolution, glm::uvec2 colsRows, uint32_t format, uint32_t timeFrameCount);
        std::vector<uint32_t> header;
        std::vector<uint8_t> packets;
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> references;
        static constexpr size_t HEADER_VALUES_COUNT{6};
        glm::uvec2 resolution()
        {
            return {header[0], header[1]};
        }
        glm::uvec2 colsRows()
        {
            return {header[2], header[3]};
        }
        Format format()
        {
            return static_cast<Format>(header[4]);
        }
        size_t timeFrameCount()
        {
            return header[5];
        }
        size_t gridSize()
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
        void init(glm::uvec2 resolution, glm::uvec2 colsRows, uint32_t format, uint32_t timeFrameCount) {data.initHeader(resolution, colsRows, format, timeFrameCount); initialized=true;};
        bool isInitialized() {return initialized;}
        void endTimeFrame(glm::uvec2 referenceCoords);

    private:
        size_t getLinearIndex(glm::ivec3 colsRowsTime);
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
        glm::uvec2 getResolution(){return data.resolution();}
        const PacketPointer getPacket(glm::ivec3 colsRowsTime);
        const PacketPointer getReferencePacket(int time);
        EncodedData data;

    private:
        std::vector<uint8_t> copyPacket(glm::ivec3 colsRowsTime);
        size_t getLinearIndex(glm::ivec3 colsRowsTime);
    };

    private:
};

