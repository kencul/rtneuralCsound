// RTNeural and standard library must be included before Csound headers to avoid
// macro collisions from Csound's C API trampling C++ reserved words.
#include <RTNeural/RTNeural.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <modload.h>
#include <plugin.h>
#include <string>
#include <unordered_map>

static constexpr float FREQ_MIN = 100.0f;
static constexpr float FREQ_MAX = 20000.0f;
static constexpr float LOG_MIN = 4.60517019f; // log(100)
static constexpr float LOG_MAX = 9.90348755f; // log(20000)

static float normalizeKnob(float freqHz) {
  float clamped = std::clamp(freqHz, FREQ_MIN, FREQ_MAX);
  return (std::log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN);
}

// Manual LayerNorm over 16 conv output features — RTNeural has no LayerNorm
// layer.
struct LayerNorm {
  static constexpr int N = 16;
  float gamma[N];
  float beta[N];

  void apply(float *x) const {
    float mean = 0.0f;
    for (int i = 0; i < N; ++i)
      mean += x[i];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; ++i) {
      float d = x[i] - mean;
      var += d * d;
    }
    float inv = 1.0f / std::sqrt(var / N + 1e-5f);

    for (int i = 0; i < N; ++i)
      x[i] = gamma[i] * (x[i] - mean) * inv + beta[i];
  }
};

// Maps knob scalar to GRU initial hidden state: Linear(1→64) + Tanh.
// Seeding h0 from the cutoff value gives the GRU prior knowledge of the target
// filter state, eliminating the cold-start transient on note-on.
struct KnobToH0 {
  static constexpr int OUT = 64;
  float weight[OUT]; // PyTorch shape [64,1] — each row is one float
  float bias[OUT];

  void compute(float knob, float (&h0)[OUT]) const {
    for (int i = 0; i < OUT; ++i)
      h0[i] = std::tanh(weight[i] * knob + bias[i]);
  }
};

// Conv stage is size-independent (1 -> 16 channels, kernel 31). The GRU/Dense
// shapes vary with hidden size H, so RecurrentStage and Model are templated.
using ConvStage =
    RTNeural::ModelT<float, 1, 16, RTNeural::Conv1DT<float, 1, 16, 31, 1>>;

template <int H>
using RecurrentStage =
    RTNeural::ModelT<float, 17, 1, RTNeural::GRULayerT<float, 17, H>,
                     RTNeural::DenseT<float, H, 1>>;

template <int H>
struct Model {
  ConvStage conv;
  LayerNorm norm;
  KnobToH0 h0net;
  RecurrentStage<H> rec;
};

// Cache parsed JSON by path — eliminates file I/O on every note-on.
// Model copy-construction is unsafe because Conv1DT uses Eigen::Map internally,
// so we load weights fresh from in-memory JSON per voice instead.
static std::unordered_map<std::string, nlohmann::json> g_json_cache;

// Fade in over this many samples at note-on to cover GRU convergence from h0.
// k2h0 seeds the GRU close to the correct state, but the model was trained with
// a 2048-sample warmup after h0 seeding, so some convergence time still exists.
static constexpr uint32_t FADE_SAMPLES = 256;

// moognn<H> aout, ain, Spath, kcutoff
// Registered under size-specific names (moognn32, moognn64, moognn128,
// moognn256) so each instance has the correct compile-time GRU/Dense shape.
// Loading a JSON whose hidden size != H would silently segfault inside
// RTNeural's loader, so callers must pick the matching opcode for their model.
template <int H>
struct MoogNN : csnd::Plugin<1, 3> {
  Model<H> *model = nullptr;
  uint32_t fade_counter = 0;

  int init() {
    fade_counter = 0;

    const char *path = inargs.str_data(1).data;

    if (g_json_cache.find(path) == g_json_cache.end()) {
      std::ifstream f(path);
      if (!f.is_open())
        return csound->init_error(std::string("moognn: cannot open model: ") +
                                  path);
      g_json_cache[path] = nlohmann::json::parse(f);
    }

    const nlohmann::json &j = g_json_cache[path];
    model = new Model<H>();

    RTNeural::torch_helpers::loadConv1D<float>(j, "conv.conv.",
                                               model->conv.get<0>());
    RTNeural::torch_helpers::loadGRU<float>(j, "gru.", model->rec.get<0>());
    RTNeural::torch_helpers::loadDense<float>(j, "dense.", model->rec.get<1>());

    // auto nw = j.at("norm.weight").get<std::vector<float>>();
    // auto nb = j.at("norm.bias").get<std::vector<float>>();
    // std::copy(nw.begin(), nw.end(), model->norm.gamma);
    // std::copy(nb.begin(), nb.end(), model->norm.beta);

    // auto h0w =
    // j.at("knob_to_h0.0.weight").get<std::vector<std::vector<float>>>(); auto
    // h0b = j.at("knob_to_h0.0.bias").get<std::vector<float>>(); for (int i =
    // 0; i < KnobToH0::OUT; ++i)
    //     model->h0net.weight[i] = h0w[i][0];
    // std::copy(h0b.begin(), h0b.end(), model->h0net.bias);

    // float knob_val = normalizeKnob((float)inargs[2]);

    // Reset to clean state.
    model->conv.reset();
    model->rec.reset();
    // float h0[KnobToH0::OUT];
    // model->h0net.compute(knob_val, h0);
    // for (int i = 0; i < KnobToH0::OUT; ++i)
    //     model->rec.get<0>().outs[i] = h0[i];

    return OK;
  }

  int deinit() {
    delete model;
    model = nullptr;
    return OK;
  }

  int aperf() {
    if (!model)
      return OK;

    MYFLT *out = outargs.data(0);
    MYFLT *in = inargs.data(0);
    float knob = normalizeKnob((float)inargs[2]);

    float convIn[1], gruIn[17];

    for (uint32_t i = offset; i < nsmps; i++) {
      convIn[0] = (float)in[i];
      model->conv.forward(convIn);

      std::copy(model->conv.getOutputs(), model->conv.getOutputs() + 16, gruIn);
      // model->norm.apply(gruIn);
      gruIn[16] = knob;

      float result = model->rec.forward(gruIn);

      float fade = (fade_counter < FADE_SAMPLES)
                       ? (float)fade_counter / FADE_SAMPLES
                       : 1.0f;
      out[i] = (MYFLT)((result + convIn[0]) * fade);
      if (fade_counter < FADE_SAMPLES)
        fade_counter++;
    }

    return OK;
  }
};

// moognn_preload Spath  — pre-warms the JSON cache so the first MIDI note-on
// doesn't pay the ~8ms parse cost. Returns 0 (dummy i-rate output required by
// CPOF — Plugin<0,N> mis-computes inarg offsets).
struct MoogNNPreload : csnd::Plugin<1, 1> {
  int init() {
    outargs[0] = 0;
    const char *path = inargs.str_data(0).data;
    if (g_json_cache.find(path) == g_json_cache.end()) {
      std::ifstream f(path);
      if (!f.is_open())
        return csound->init_error(std::string("moognn_preload: cannot open: ") +
                                  path);
      g_json_cache[path] = nlohmann::json::parse(f);
      csound->message(std::string("moognn_preload: cached ") + path + "\n");
    }
    return OK;
  }
};

void csnd::on_load(csnd::Csound *csound) {
  csnd::plugin<MoogNN<32>>(csound, "moognn32", "a", "aSk", csnd::thread::ia);
  csnd::plugin<MoogNN<64>>(csound, "moognn64", "a", "aSk", csnd::thread::ia);
  csnd::plugin<MoogNN<128>>(csound, "moognn128", "a", "aSk", csnd::thread::ia);
  csnd::plugin<MoogNN<256>>(csound, "moognn256", "a", "aSk", csnd::thread::ia);
  csnd::plugin<MoogNNPreload>(csound, "moognn_preload", "i", "S",
                              csnd::thread::i);
}
