// Process a WAV file sample-by-sample using the conditioned filter model from tensor_torch_param.py.
// Architecture: CausalConv1d(1,16,31) -> LayerNorm(16) -> GRU(17,128) -> Dense(128,1)
// The knob (normalized cutoff frequency) seeds the GRU hidden state via a learned Linear+Tanh mapping.

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <RTNeural/RTNeural.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

static constexpr float FREQ_MIN = 60.0f;
static constexpr float FREQ_MAX = 20000.0f;
static constexpr float LOG_MIN  = 4.09434456f; // std::log(60)
static constexpr float LOG_MAX  = 9.90348755f; // std::log(20000)

static float normalizeKnob(float freqHz)
{
    float clamped = std::clamp(freqHz, FREQ_MIN, FREQ_MAX);
    return (std::log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN);
}

// Manual LayerNorm over 16 features — RTNeural has no LayerNorm layer
struct LayerNorm
{
    static constexpr int N = 16;
    float gamma[N];
    float beta[N];

    void apply(float* x) const
    {
        float mean = 0.0f;
        for(int i = 0; i < N; ++i) mean += x[i];
        mean /= N;

        float var = 0.0f;
        for(int i = 0; i < N; ++i) { float d = x[i] - mean; var += d * d; }
        float inv = 1.0f / std::sqrt(var / N + 1e-5f);

        for(int i = 0; i < N; ++i)
            x[i] = gamma[i] * (x[i] - mean) * inv + beta[i];
    }
};

// knob_to_h0: Linear(1,128) + Tanh, maps knob scalar to GRU initial hidden state
struct KnobToH0
{
    static constexpr int OUT = 128;
    float weight[OUT]; // weight shape is [128,1] in PyTorch — each row is one float
    float bias[OUT];

    void compute(float knob, float (&h0)[OUT]) const
    {
        for(int i = 0; i < OUT; ++i)
            h0[i] = std::tanh(weight[i] * knob + bias[i]);
    }
};

// Conv processes audio only (1 channel in, 16 out, kernel 31)
using ConvStage = RTNeural::ModelT<float, 1, 16, RTNeural::Conv1DT<float, 1, 16, 31, 1>>;

// GRU takes 17 inputs (16 conv features + 1 knob), Dense maps 128 -> 1
using RecurrentStage = RTNeural::ModelT<float, 17, 1,
    RTNeural::GRULayerT<float, 17, 128>,
    RTNeural::DenseT<float, 128, 1>>;

struct Model
{
    ConvStage      conv;
    LayerNorm      norm;
    KnobToH0       h0net;
    RecurrentStage rec;
};

static void loadWeights(const std::string& path, Model& m)
{
    std::ifstream f(path, std::ifstream::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open model file: " + path);

    nlohmann::json j;
    f >> j;

    // Keys match PyTorch state_dict attribute paths from tensor_torch_param.py
    RTNeural::torch_helpers::loadConv1D<float>(j, "conv.conv.", m.conv.get<0>());
    RTNeural::torch_helpers::loadGRU<float>(j, "gru.", m.rec.get<0>());
    RTNeural::torch_helpers::loadDense<float>(j, "dense.", m.rec.get<1>());

    auto nw = j.at("norm.weight").get<std::vector<float>>();
    auto nb = j.at("norm.bias").get<std::vector<float>>();
    std::copy(nw.begin(), nw.end(), m.norm.gamma);
    std::copy(nb.begin(), nb.end(), m.norm.beta);

    // shape [128, 1] in PyTorch — each row is a single float
    auto h0w = j.at("knob_to_h0.0.weight").get<std::vector<std::vector<float>>>();
    auto h0b = j.at("knob_to_h0.0.bias").get<std::vector<float>>();
    for(int i = 0; i < KnobToH0::OUT; ++i)
        m.h0net.weight[i] = h0w[i][0];
    std::copy(h0b.begin(), h0b.end(), m.h0net.bias);
}

// Reset conv state and seed GRU hidden state from the knob value before processing a file
static void prepareModel(Model& m, float knob)
{
    m.conv.reset();
    m.rec.reset();

    float h0[KnobToH0::OUT];
    m.h0net.compute(knob, h0);

    // GRULayerT exposes 'outs' as the live hidden state
    for(int i = 0; i < KnobToH0::OUT; ++i)
        m.rec.get<0>().outs[i] = h0[i];
}

static float processSample(Model& m, float audio, float knob)
{
    const float audioIn[1] = { audio };
    m.conv.forward(audioIn);

    float features[16];
    std::copy(m.conv.getOutputs(), m.conv.getOutputs() + 16, features);
    m.norm.apply(features);

    float gruIn[17];
    std::copy(features, features + 16, gruIn);
    gruIn[16] = knob;

    return m.rec.forward(gruIn);
}

int main(int argc, char* argv[])
{
    if(argc != 5)
    {
        std::cout << "Usage: process_wav_torch_param2 <model.json> <input.wav> <output.wav> <cutoff_hz>\n"
                  << "  cutoff_hz: filter cutoff in Hz (" << FREQ_MIN << " - " << FREQ_MAX << ")\n";
        return 1;
    }

    float cutoffHz = std::stof(argv[4]);
    float knob     = normalizeKnob(cutoffHz);
    std::cout << "Cutoff: " << cutoffHz << " Hz  (knob=" << knob << ")\n";

    // Heap-allocate: Eigen fixed-size matrices for GRU(17,128) exceed default stack limits
    auto mL = std::make_unique<Model>();
    auto mR = std::make_unique<Model>();

    try
    {
        loadWeights(argv[1], *mL);
        loadWeights(argv[1], *mR);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    drwav wav;
    if(!drwav_init_file(&wav, argv[2], nullptr))
    {
        std::cerr << "Failed to open input WAV: " << argv[2] << '\n';
        return 1;
    }

    std::vector<float> samples(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, samples.data());
    const uint32_t sampleRate = wav.sampleRate;
    const uint32_t channels   = wav.channels;
    const uint64_t frameCount = wav.totalPCMFrameCount;
    drwav_uninit(&wav);

    prepareModel(*mL, knob);
    prepareModel(*mR, knob);

    // Output is always stereo; mono input is duplicated to both channels
    std::vector<float> output(frameCount * 2);
    for(uint64_t i = 0; i < frameCount; ++i)
    {
        float sL = samples[i * channels];
        float sR = channels > 1 ? samples[i * channels + 1] : sL;
        output[i * 2]     = processSample(*mL, sL, knob);
        output[i * 2 + 1] = processSample(*mR, sR, knob);
    }

    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = 2;
    fmt.sampleRate    = sampleRate;
    fmt.bitsPerSample = 32;

    drwav outWav;
    if(!drwav_init_file_write(&outWav, argv[3], &fmt, nullptr))
    {
        std::cerr << "Failed to open output WAV: " << argv[3] << '\n';
        return 1;
    }
    drwav_write_pcm_frames(&outWav, frameCount, output.data());
    drwav_uninit(&outWav);

    std::cout << "Wrote " << frameCount << " frames (stereo) to " << argv[3] << '\n';
    return 0;
}