#include <fstream>
#include <string>
#include <iostream>
#include "muxing.h"

const std::set<std::filesystem::path> Muxing::listPath(std::string path)
{
    auto dir = std::filesystem::directory_iterator(path);
    std::set<std::filesystem::path> sorted;
    for(const auto &file : dir)
        sorted.insert(file.path().filename());
    return sorted;
}

glm::uvec2 Muxing::parseFilename(std::string name)
{
    int delimiterPos = name.find('_');
    int extensionPos = name.find('.');
    auto row = name.substr(0, delimiterPos);
    auto col = name.substr(delimiterPos + 1, extensionPos - delimiterPos - 1);
    return {stoi(row), stoi(col)};
}

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
    constexpr size_t BYTE_COUNT{4};
    fos.write(reinterpret_cast<const char *>(data.header.data()), data.header.size()*BYTE_COUNT);
    fos.write(reinterpret_cast<const char *>(data.offsets.data()), data.offsets.size()*BYTE_COUNT);
    fos.write(reinterpret_cast<const char *>(data.packets.data()), data.packets.size()*BYTE_COUNT);
    fos.close();
}

void Muxing::Muxer::addPacket(const std::vector<uint8_t> *packetData)
{
    data.addData(packetData);
}

Muxing::Demuxer::Demuxer(std::string filePath)
{
    std::cout << "Loading LF data..." << std::endl;
    std::ifstream fis(filePath, std::ios::binary);
    constexpr size_t BYTE_COUNT{4};
    data.header.resize(EncodedData::HEADER_VALUES_COUNT);
    fis.read(reinterpret_cast<char *>(data.header.data()), BYTE_COUNT*EncodedData::HEADER_VALUES_COUNT);

    data.offsets.resize(data.frameCount());
    fis.read(reinterpret_cast<char *>(data.offsets.data()), data.offsets.size()*BYTE_COUNT);
    data.packets.resize(data.offsets.back());
    fis.read(reinterpret_cast<char *>(data.packets.data()), data.offsets.back());
}

std::vector<uint8_t> Muxing::Demuxer::getPacket(size_t index)
{
    size_t start = data.offsets[index];
    size_t end = data.offsets[index + 1];
    return std::vector<uint8_t>(data.packets.begin() + start, data.packets.begin() + end);
}
