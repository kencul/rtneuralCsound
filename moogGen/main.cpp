#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

#include "example/helpers.hpp"
#include "src/RKSimulationModel.h"

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputDir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            inputFile = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputDir = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " -f <input.wav> [-o <output_dir>]\n";
            return 1;
        }
    }

    if (inputFile.empty()) {
        std::cerr << "Usage: " << argv[0] << " -f <input.wav> [-o <output_dir>]\n";
        return 1;
    }

    if (!outputDir.empty())
        std::filesystem::create_directories(outputDir);

    int sampleRate = 0;
    int numChannels = 0;
    std::vector<float> inputSamples;

    // Load file completely into memory to prevent disk I/O bottlenecks during processing
    if (!ReadWavFile(inputFile.c_str(), sampleRate, numChannels, inputSamples)) {
        std::cerr << "Failed to load " << inputFile << "\n";
        return 1;
    }

    // Strip directory and extension to use as output prefix (e.g. "audio/bench.wav" -> "bench")
    std::string stem = inputFile;
    auto slash = stem.find_last_of("/\\");
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    auto dot = stem.rfind('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);

    std::vector<float> cutoffs = {20.0f, 60.0f, 100.0f, 125.0f, 250.0f, 500.0f, 800.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f, 12000.0f, 16000.0f, 20000.0f, 24000.0f};
    float resonance = 0.5f;

    for (float cutoff : cutoffs) {
        RKSimulationMoog filter(static_cast<float>(sampleRate));
        filter.SetCutoff(cutoff);
        filter.SetResonance(resonance);

        // A fresh copy of the audio is allocated because the Process function modifies the buffer in place
        std::vector<float> processingBuffer = inputSamples;

        filter.Process(processingBuffer.data(), static_cast<uint32_t>(processingBuffer.size()));

        std::string outputFile = (outputDir.empty() ? "" : outputDir + "/") + stem + "_" + std::to_string(static_cast<int>(cutoff)) + "hz.wav";

        if (WriteWavFile(outputFile.c_str(), sampleRate, numChannels, processingBuffer)) {
            std::cout << "Successfully generated: " << outputFile << "\n";
        } else {
            std::cerr << "Failed to write: " << outputFile << "\n";
        }
    }

    return 0;
}