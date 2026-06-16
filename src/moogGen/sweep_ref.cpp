// Runs the RK Moog ladder filter with a time-varying cutoff sweep and writes
// the filtered audio to a WAV and the per-sample cutoff schedule to a CSV.
// The CSV is consumed by eval_dynamic.py to drive the neural model with the
// exact same cutoff trajectory.
//
// Usage:
//   sweep_ref <in.wav> <out.wav> <out.csv> log <freq_start> <freq_end> [resonance=1.0]
//   sweep_ref <in.wav> <out.wav> <out.csv> lfo <freq_low> <freq_high> [resonance=1.0] [lfo_rate_hz=1.0]

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <RKSimulationModel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Must match normalizeKnob() in moognn.cpp and eval_param_model.py
static constexpr float KNOB_FREQ_MIN = 100.0f;
static constexpr float KNOB_FREQ_MAX = 20000.0f;
static const float KNOB_LOG_MIN = std::log(KNOB_FREQ_MIN);
static const float KNOB_LOG_MAX = std::log(KNOB_FREQ_MAX);

static float normalizeKnob(float freqHz)
{
    float clamped = std::clamp(freqHz, KNOB_FREQ_MIN, KNOB_FREQ_MAX);
    return (std::log(clamped) - KNOB_LOG_MIN) / (KNOB_LOG_MAX - KNOB_LOG_MIN);
}

// Exponential ramp from freqStart to freqEnd over totalFrames samples
static float logSweepFreq(uint64_t i, uint64_t totalFrames, float freqStart, float freqEnd)
{
    double t = static_cast<double>(i) / static_cast<double>(totalFrames - 1);
    return static_cast<float>(freqStart * std::pow(static_cast<double>(freqEnd) / freqStart, t));
}

// Sinusoidal LFO in log-frequency space, oscillating between freqLow and freqHigh
static float lfoSweepFreq(uint64_t i, float sampleRate, float freqLow, float freqHigh, float lfoRate)
{
    double logMid   = 0.5 * (std::log(freqLow) + std::log(freqHigh));
    double logRange = 0.5 * (std::log(freqHigh) - std::log(freqLow));
    double phase    = 2.0 * MOOG_PI * lfoRate * i / sampleRate;
    return static_cast<float>(std::exp(logMid + logRange * std::sin(phase)));
}

int main(int argc, char* argv[])
{
    if (argc < 7) {
        std::cerr << "Usage:\n"
                  << "  sweep_ref <in.wav> <out.wav> <out.csv> log <freq_start> <freq_end> [resonance=1.0]\n"
                  << "  sweep_ref <in.wav> <out.wav> <out.csv> lfo <freq_low> <freq_high> [resonance=1.0] [lfo_rate_hz=1.0]\n";
        return 1;
    }

    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];
    const char* csvPath    = argv[3];
    const char* sweepArg   = argv[4];
    float freqA     = std::stof(argv[5]);
    float freqB     = std::stof(argv[6]);
    float resonance = argc > 7 ? std::stof(argv[7]) : 1.0f;
    float lfoRate   = argc > 8 ? std::stof(argv[8]) : 1.0f;

    bool isLog = std::strcmp(sweepArg, "log") == 0;
    bool isLFO = std::strcmp(sweepArg, "lfo") == 0;
    if (!isLog && !isLFO) {
        std::cerr << "Unknown sweep type '" << sweepArg << "'. Use 'log' or 'lfo'.\n";
        return 1;
    }

    drwav wav;
    if (!drwav_init_file(&wav, inputPath, nullptr)) {
        std::cerr << "Failed to open input WAV: " << inputPath << '\n';
        return 1;
    }

    const uint32_t sampleRate = wav.sampleRate;
    const uint32_t channels   = wav.channels;
    const uint64_t frameCount = wav.totalPCMFrameCount;

    std::vector<float> raw(frameCount * channels);
    drwav_read_pcm_frames_f32(&wav, frameCount, raw.data());
    drwav_uninit(&wav);

    // Mix to mono
    std::vector<float> mono(frameCount);
    for (uint64_t i = 0; i < frameCount; ++i) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < channels; ++c)
            sum += raw[i * channels + c];
        mono[i] = sum / static_cast<float>(channels);
    }

    RKSimulationMoog moog(static_cast<float>(sampleRate));
    moog.SetResonance(resonance);

    std::vector<float> output(frameCount);
    std::vector<float> schedule(frameCount);

    for (uint64_t i = 0; i < frameCount; ++i) {
        float freq = isLog
            ? logSweepFreq(i, frameCount, freqA, freqB)
            : lfoSweepFreq(i, static_cast<float>(sampleRate), freqA, freqB, lfoRate);

        schedule[i] = normalizeKnob(freq);
        moog.SetCutoff(freq);
        float s = mono[i];
        moog.Process(&s, 1);
        output[i] = s;
    }

    // Write float32 WAV so the Python script can load without quantization loss
    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = 1;
    fmt.sampleRate    = sampleRate;
    fmt.bitsPerSample = 32;

    drwav outWav;
    if (!drwav_init_file_write(&outWav, outputPath, &fmt, nullptr)) {
        std::cerr << "Failed to open output WAV: " << outputPath << '\n';
        return 1;
    }
    drwav_write_pcm_frames(&outWav, frameCount, output.data());
    drwav_uninit(&outWav);

    // Write per-sample cutoff schedule — Python reads this to drive the neural model
    FILE* csv = std::fopen(csvPath, "w");
    if (!csv) {
        std::cerr << "Failed to open CSV: " << csvPath << '\n';
        return 1;
    }
    std::fprintf(csv, "sample,knob\n");
    for (uint64_t i = 0; i < frameCount; ++i)
        std::fprintf(csv, "%llu,%.8f\n", static_cast<unsigned long long>(i), schedule[i]);
    std::fclose(csv);

    std::cout << "Wrote " << frameCount << " samples to " << outputPath
              << " and schedule to " << csvPath << '\n';
    return 0;
}
