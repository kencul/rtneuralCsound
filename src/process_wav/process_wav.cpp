#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <RTNeural/RTNeural.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage: process_wav <model.json> <input.wav> <output.wav>" << std::endl;
        return 1;
    }

    std::ifstream jsonStream(argv[1], std::ifstream::binary);
    if (!jsonStream.is_open())
    {
        std::cerr << "Failed to open model: " << argv[1] << std::endl;
        return 1;
    }
    auto model = RTNeural::json_parser::parseJson<float>(jsonStream, true);

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

    // Mix down to mono if needed, then process sample by sample
    std::vector<float> mono(frameCount);
    for (uint64_t i = 0; i < frameCount; ++i)
    {
        float sum = 0.0f;
        for (uint32_t c = 0; c < channels; ++c)
            sum += samples[i * channels + c];
        mono[i] = sum / static_cast<float>(channels);
    }

    model->reset();
    std::vector<float> output(frameCount);
    for (uint64_t i = 0; i < frameCount; ++i)
        output[i] = model->forward(&mono[i]);

    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = 1;
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

    std::cout << "Wrote " << frameCount << " samples to " << argv[3] << std::endl;
    return 0;
}
