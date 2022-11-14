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
        void initHeader(glm::uvec2 resolution, glm::uvec2 colsRows, glm::uvec2 reference, uint32_t format);
        std::vector<uint32_t> header;
        std::vector<uint8_t> packets;
        std::vector<uint32_t> offsets;
        static constexpr size_t HEADER_VALUES_COUNT{7};
        glm::uvec2 resolution()
        {
            return {header[0], header[1]};
        }
        glm::uvec2 colsRows()
        {
            return {header[2], header[3]};
        }
        glm::uvec2 referencePosition()
        {
            return {header[4], header[5]};
        }
        Format format()
        {
            return static_cast<Format>(header[6]);
        }
        size_t frameCount()
        {
            return colsRows().x * colsRows().y;
        }
    };

    class Muxer
    {
    public:
        Muxer(glm::uvec2 resolution, glm::uvec2 colsRows, glm::uvec2 reference, uint32_t format)
        {
            data.initHeader(resolution, colsRows, reference, format);
        };
        friend void operator<<(Muxer &m, const std::vector<uint8_t> *packet)
        {
            m.addPacket(packet);
        };
        void save(std::string filePath);

    private:
        void addPacket(const std::vector<uint8_t> *packetData);
        EncodedData data;
    };

    class Demuxer
    {
    public:
        Demuxer(std::string filePath);
        friend void operator<<(std::vector<uint8_t> &packet, Demuxer &m)
        {
            packet = m.getPacket(m.exportedPacketCount);
            m.exportedPacketCount++;
        };
        EncodedData data;

    private:
        size_t exportedPacketCount{0};
        std::vector<uint8_t> getPacket(size_t index);
    };

private:
};

