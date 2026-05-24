#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>


inline bool ReadWavFile(const char* filename, int& sampleRate, int& numChannels, std::vector<float>& samples)
{
    std::ifstream file(filename, std::ios::binary);
    if(!file)
        return false;

    auto read32 = [&]()
    { uint32_t v; file.read((char*)&v, 4); return v; };
    auto read16 = [&]()
    { uint16_t v; file.read((char*)&v, 2); return v; };

    char riff[4], wave[4];
    file.read(riff, 4);
    read32();
    file.read(wave, 4);
    if(std::memcmp(riff, "RIFF", 4) || std::memcmp(wave, "WAVE", 4))
        return false;

    uint16_t audioFormat = 0, bitsPerSample = 0;
    uint32_t dataSize = 0;
    bool foundFmt = false, foundData = false;

    while(file && !(foundFmt && foundData))
    {
        char chunkId[4];
        file.read(chunkId, 4);
        uint32_t chunkSize = read32();
        std::streampos chunkStart = file.tellg();

        if(std::memcmp(chunkId, "fmt ", 4) == 0)
        {
            audioFormat = read16();
            numChannels = read16();
            sampleRate = (int)read32();
            read32();
            read16();
            bitsPerSample = read16();
            foundFmt = true;
        }
        else if(std::memcmp(chunkId, "data", 4) == 0)
        {
            dataSize = chunkSize;
            foundData = true;
            break;
        }
        file.seekg(chunkStart + (std::streamoff)chunkSize);
    }

    if(!foundFmt || !foundData || (audioFormat != 1 && audioFormat != 3))
        return false;

    uint32_t numSamples = dataSize / (bitsPerSample / 8);
    samples.resize(numSamples);

    if(audioFormat == 3)
    {
        file.read((char*)samples.data(), dataSize);
    }
    else if(bitsPerSample == 16)
    {
        std::vector<int16_t> raw(numSamples);
        file.read((char*)raw.data(), dataSize);
        for(uint32_t i = 0; i < numSamples; i++)
            samples[i] = raw[i] / 32768.0f;
    }
    else if(bitsPerSample == 24)
    {
        for(uint32_t i = 0; i < numSamples; i++)
        {
            uint8_t b[3];
            file.read((char*)b, 3);
            int32_t v = (b[0] << 8) | (b[1] << 16) | (b[2] << 24);
            samples[i] = (v >> 8) / 8388608.0f;
        }
    }
    else if(bitsPerSample == 32)
    {
        std::vector<int32_t> raw(numSamples);
        file.read((char*)raw.data(), dataSize);
        for(uint32_t i = 0; i < numSamples; i++)
            samples[i] = raw[i] / 2147483648.0f;
    }
    else if(bitsPerSample == 8)
    {
        std::vector<uint8_t> raw(numSamples);
        file.read((char*)raw.data(), dataSize);
        for(uint32_t i = 0; i < numSamples; i++)
            samples[i] = (raw[i] - 128) / 128.0f;
    }
    else
    {
        return false;
    }

    return file.good() || file.eof();
}

inline bool WriteWavFile(const char* filename, int sampleRate, int numChannels, const std::vector<float>& samples)
{
    std::ofstream file(filename, std::ios::binary);
    if(!file)
        return false;

    auto write32 = [&](uint32_t v)
    { file.write((char*)&v, 4); };
    auto write16 = [&](uint16_t v)
    { file.write((char*)&v, 2); };

    uint32_t numSamples = static_cast<uint32_t>(samples.size());
    uint16_t bitsPerSample = 16;
    uint32_t dataSize = numSamples * (bitsPerSample / 8);
    uint32_t fileSize = 36 + dataSize;

    file.write("RIFF", 4);
    write32(fileSize);
    file.write("WAVE", 4);

    file.write("fmt ", 4);
    write32(16);
    write16(1);
    write16(static_cast<uint16_t>(numChannels));
    write32(static_cast<uint32_t>(sampleRate));
    write32(static_cast<uint32_t>(sampleRate * numChannels * (bitsPerSample / 8)));
    write16(static_cast<uint16_t>(numChannels * (bitsPerSample / 8)));
    write16(bitsPerSample);

    file.write("data", 4);
    write32(dataSize);

    for(uint32_t i = 0; i < numSamples; i++)
    {
        float clamped = samples[i];
        if(clamped > 1.0f)
            clamped = 1.0f;
        if(clamped < -1.0f)
            clamped = -1.0f;
        int16_t sample = static_cast<int16_t>(clamped * 32767.0f);
        file.write((char*)&sample, 2);
    }

    return file.good();
}