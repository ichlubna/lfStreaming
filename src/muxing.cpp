#include <fstream>
#include <string>
#include "muxing.h"

void Muxing::EncodedData::addData(const std::vector<uint8_t> *packetData)
{
    offsets.push_back(packets.size());
    packets.insert(packets.end(), packetData->begin(), packetData->end());
}

void Muxing::EncodedData::initHeader(glm::uvec2 resolution, glm::uvec2 rowsCols, glm::uvec2 reference, uint32_t format)
{
    header = {resolution.x, resolution.y, rowsCols.x, rowsCols.y, reference.x, reference.y, format};
    packets.reserve(rowsCols.x * rowsCols.y);
    packets.push_back(0);
}

void Muxing::Muxer::save(std::string filePath)
{
    data.offsets.push_back(data.packets.size());
    std::ofstream fos(filePath, std::ios::binary);
    fos.write(reinterpret_cast<const char *>(data.header.data()), data.header.size());
    fos.write(reinterpret_cast<const char *>(data.offsets.data()), data.offsets.size());
    fos.write(reinterpret_cast<const char *>(data.packets.data()), data.packets.size());
    fos.close();
}

void Muxing::Muxer::addPacket(const std::vector<uint8_t> *packetData)
{
    data.addData(packetData);
}

Muxing::Demuxer::Demuxer(std::string filePath)
{
    std::ifstream fis(filePath, std::ios::binary);
    constexpr size_t BYTE_COUNT{4};
    data.header.resize(EncodedData::HEADER_VALUES_COUNT);
    for(auto &headerValue : data.header)
        fis.read(reinterpret_cast<char *>(&headerValue), BYTE_COUNT);

    data.offsets.resize(data.frameCount());
    fis.read(reinterpret_cast<char *>(data.offsets.data()), data.offsets.size()*BYTE_COUNT);
    fis.read(reinterpret_cast<char *>(data.packets.data()), data.offsets.back());
}

std::vector<uint8_t> Muxing::Demuxer::getPacket(size_t index)
{
    size_t start = data.offsets[index];
    size_t end = data.offsets[index + 1];
    return std::vector<uint8_t>(data.packets.begin() + start, data.packets.begin() + end);
}
