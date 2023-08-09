#include <fstream>
#include <stdexcept>
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
    auto delimiterPos = name.find('_');
    if(delimiterPos == std::string::npos)
        throw std::runtime_error("Missing delimiter in " + name + " as column_row!");
    int extensionPos = name.find('.');
    auto row = name.substr(0, delimiterPos);
    auto col = name.substr(delimiterPos + 1, extensionPos - delimiterPos - 1);
    return {stoi(row), stoi(col)};
}

size_t Muxing::Muxer::getLinearIndex(glm::ivec3 colsRowsTime) const
{
    return data.gridSize() * colsRowsTime.z + colsRowsTime.y * data.colsRows().x + colsRowsTime.x;
}

void Muxing::Muxer::endTimeFrame(glm::uvec2 referenceCoords)
{
    unsigned int referenceID = getLinearIndex({referenceCoords, frameNumber});
    data.references.push_back(referenceID);
    frameNumber++;
}

void Muxing::EncodedData::addData(const std::vector<uint8_t> *packetData)
{
    offsets.push_back(packets.size());
    packets.insert(packets.end(), packetData->begin(), packetData->end());
}

void Muxing::EncodedData::initHeader(glm::uvec2 resolution, glm::uvec2 colsRows, uint32_t format, uint32_t timeFrameCount, float aspect)
{
    header = {resolution.x, resolution.y, colsRows.x, colsRows.y, format, timeFrameCount, static_cast<unsigned int>(aspect*Muxing::EncodedData::FIXED_FLOAT_MULTIPLIER)};
    size_t count{colsRows.x *colsRows.y * timeFrameCount};
    offsets.reserve(count);
    references.reserve(count);
}

void Muxing::Muxer::save(std::string filePath)
{
    data.offsets.push_back(data.packets.size());
    std::ofstream fos(filePath, std::ios::binary);
    constexpr size_t BYTE_COUNT{4};
    fos.write(reinterpret_cast<const char *>(data.header.data()), data.header.size()*BYTE_COUNT);
    fos.write(reinterpret_cast<const char *>(data.offsets.data()), data.offsets.size()*BYTE_COUNT);
    fos.write(reinterpret_cast<const char *>(data.references.data()), data.references.size()*BYTE_COUNT);
    fos.write(reinterpret_cast<const char *>(data.packets.data()), data.packets.size());
    //for(auto o : data.offsets) std::cerr << o << " ";
    //std::cerr << std::endl;
    //std::cerr << data.header.size() << " " << data.offsets.size() << " " << data.references.size() << " " << data.packets.size() << std::endl;
    fos.close();
}

void Muxing::Muxer::addPacket(const std::vector<uint8_t> *packetData)
{
    data.addData(packetData);
}

Muxing::Demuxer::Demuxer(std::string filePath)
{
    std::cout << "Loading LF data..." << std::endl;
    std::ifstream fis(filePath, std::ios::binary | std::ios::ate);
    size_t fileSize{static_cast<size_t>(fis.tellg())};
    fis.clear();
    fis.seekg(0, std::ios::beg);

    if(fileSize < EncodedData::HEADER_VALUES_COUNT)
        throw std::runtime_error("Input file is empty.");

    constexpr size_t BYTE_COUNT{4};
    data.header.resize(EncodedData::HEADER_VALUES_COUNT);
    fis.read(reinterpret_cast<char *>(data.header.data()), BYTE_COUNT * EncodedData::HEADER_VALUES_COUNT);

    data.offsets.resize(1 + data.gridSize()*data.timeFrameCount());
    fis.read(reinterpret_cast<char *>(data.offsets.data()), data.offsets.size()*BYTE_COUNT);
    data.references.resize(data.timeFrameCount());
    fis.read(reinterpret_cast<char *>(data.references.data()), data.references.size()*BYTE_COUNT);

    size_t calculatedFileSize{BYTE_COUNT *(data.gridSize() + data.timeFrameCount() + EncodedData::HEADER_VALUES_COUNT) + data.offsets.back()};
    if(fileSize < calculatedFileSize)
        throw std::runtime_error("Missing packets data in the input file.");

    data.packets.resize(data.offsets.back());
    fis.read(reinterpret_cast<char *>(data.packets.data()), data.offsets.back());
    //std::cerr << EncodedData::HEADER_VALUES_COUNT << " " << 1+data.gridSize()*data.timeFrameCount() << " " << data.timeFrameCount() << " " << data.packets.size();
}

size_t Muxing::Demuxer::getLinearIndex(glm::ivec3 colsRowsTime) const
{
    return data.gridSize() * colsRowsTime.z + colsRowsTime.y * data.colsRows().x + colsRowsTime.x;
}

std::vector<uint8_t> Muxing::Demuxer::copyPacket(glm::ivec3 colsRowsTime) const
{
    size_t index = getLinearIndex(colsRowsTime);
    size_t start = data.offsets[index];
    size_t end = data.offsets[index + 1];
    return std::vector<uint8_t>(data.packets.begin() + start, data.packets.begin() + end);
}

const Muxing::Demuxer::PacketPointer Muxing::Demuxer::getReferencePacket(int time) const
{
    PacketPointer packetPointer;
    size_t referenceID = data.references[time];
    packetPointer.data = &data.packets.data()[data.offsets[referenceID]];
    packetPointer.size = data.offsets[referenceID + 1] - data.offsets[referenceID];
    return packetPointer;
}

const Muxing::Demuxer::PacketPointer Muxing::Demuxer::getPacket(glm::ivec3 colsRowsTime) const
{
    PacketPointer packetPointer;
    size_t index = getLinearIndex(colsRowsTime);
    packetPointer.data = &data.packets.data()[data.offsets[index]];
    packetPointer.size = data.offsets[index + 1] - data.offsets[index];
    return packetPointer;
}
