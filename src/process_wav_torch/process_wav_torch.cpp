#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <RTNeural/RTNeural.h>
#include <iostream>
#include <fstream>
#include <vector>

// Matches tensor_torch.py: CausalConv1d(1,16,31) -> GRU(16,64) -> Linear(64,1)
using ModelType = RTNeural::ModelT<float, 1, 1,
    RTNeural::Conv1DT<float, 1, 16, 31, 1>,
    RTNeural::GRULayerT<float, 16, 64>,
    RTNeural::DenseT<float, 64, 1>
>;

void loadModel(const std::string& path, ModelType& model)
{
    std::ifstream jsonStream(path, std::ifstream::binary);
    if (!jsonStream.is_open())
        throw std::runtime_error("Failed to open model: " + path);

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    // Key prefixes match PyTorch state_dict attribute paths from tensor_torch.py:
    // self.conv.conv -> "conv.conv.", self.gru -> "gru.", self.dense -> "dense."
    RTNeural::torch_helpers::loadConv1D<float>(modelJson, "conv.conv.", model.get<0>());
    RTNeural::torch_helpers::loadGRU<float>(modelJson, "gru.", model.get<1>());
    RTNeural::torch_helpers::loadDense<float>(modelJson, "dense.", model.get<2>());
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage: process_wav_torch <model.json> <input.wav> <output.wav>" << std::endl;
        return 1;
    }

    // Two instances share the same weights but maintain independent filter state per channel
    ModelType modelL, modelR;
    try { loadModel(argv[1], modelL); loadModel(argv[1], modelR); }
    catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }

    drwav wav;
    if (!drwav_init_file(&wav, argv[2], nullptr))
    {
        std::cerr << "Failed to open WAV: " << argv[2] << std::endl;
        return 1;
    }

    std::vector<float> samples(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, samples.data());

    uint32_t sampleRate = wav.sampleRate;
    uint32_t channels   = wav.channels;
    uint64_t frameCount = wav.totalPCMFrameCount;
    drwav_uninit(&wav);

    modelL.reset();
    modelR.reset();

    // Output is always stereo — mono input is duplicated to both channels
    std::vector<float> output(frameCount * 2);
    for (uint64_t i = 0; i < frameCount; ++i)
    {
        float left  = samples[i * channels];
        float right = channels > 1 ? samples[i * channels + 1] : left;
        output[i * 2]     = modelL.forward(&left);
        output[i * 2 + 1] = modelR.forward(&right);
    }

    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = 2;
    fmt.sampleRate    = sampleRate;
    fmt.bitsPerSample = 32;

    drwav outWav;
    if (!drwav_init_file_write(&outWav, argv[3], &fmt, nullptr))
    {
        std::cerr << "Failed to open output WAV: " << argv[3] << std::endl;
        return 1;
    }
    drwav_write_pcm_frames(&outWav, frameCount, output.data());
    drwav_uninit(&outWav);

    std::cout << "Wrote " << frameCount << " frames (stereo) to " << argv[3] << std::endl;
    return 0;
}
