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

using ConvStage =
    RTNeural::ModelT<float, 1, 16, RTNeural::Conv1DT<float, 1, 16, 31, 1>>;

// 16 conv features in (no knob channel), GRU 128 units, Dense 128→1
using RecurrentStage =
    RTNeural::ModelT<float, 16, 1, RTNeural::GRULayerT<float, 16, 128>,
                     RTNeural::DenseT<float, 128, 1>>;

struct Model {
  ConvStage conv;
  RecurrentStage rec;
};

// Cache parsed JSON by path — avoids re-parsing on every note-on.
static std::unordered_map<std::string, nlohmann::json> g_json_cache;

// Fade-in at note-on: covers GRU cold-start convergence from zero hidden state.
static constexpr uint32_t FADE_SAMPLES = 256;

// distnn aout, ain, Spath, kmix
// ain:   audio input
// Spath: path to weights.json exported by tensor_torch_distortion.py
// kmix:  wet/dry blend (0 = dry, 1 = wet), k-rate
struct DistNN : csnd::Plugin<1, 3> {
  Model *model = nullptr;
  uint32_t fade_counter = 0;

  int init() {
    fade_counter = 0;

    const char *path = inargs.str_data(1).data;

    if (g_json_cache.find(path) == g_json_cache.end()) {
      std::ifstream f(path);
      if (!f.is_open())
        return csound->init_error(std::string("distnn: cannot open model: ") +
                                  path);
      g_json_cache[path] = nlohmann::json::parse(f);
    }

    const nlohmann::json &j = g_json_cache[path];
    model = new Model();

    // dist_05 weights were exported before model_distortion.py renamed self.gru
    // to self.rnn, so JSON keys use the "gru." prefix. eval_distortion.py remaps
    // this transparently; the C++ loader must match the actual JSON key prefix.
    RTNeural::torch_helpers::loadConv1D<float>(j, "conv.conv.",
                                               model->conv.get<0>());
    RTNeural::torch_helpers::loadGRU<float>(j, "gru.", model->rec.get<0>());
    RTNeural::torch_helpers::loadDense<float>(j, "dense.", model->rec.get<1>());

    model->conv.reset();
    model->rec.reset();

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
    float mix = std::clamp((float)inargs[2], 0.0f, 1.0f);

    float convIn[1], gruIn[16];

    for (uint32_t i = offset; i < nsmps; i++) {
      convIn[0] = (float)in[i];
      model->conv.forward(convIn);

      // Conv1DT streams sample-by-sample and maintains its own 30-sample delay
      // line internally — no explicit context buffer needed across ksmps blocks.
      std::copy(model->conv.getOutputs(), model->conv.getOutputs() + 16,
                gruIn);

      float wet = model->rec.forward(gruIn) + convIn[0]; // skip connection

      float fade = (fade_counter < FADE_SAMPLES)
                       ? (float)fade_counter / FADE_SAMPLES
                       : 1.0f;
      if (fade_counter < FADE_SAMPLES)
        fade_counter++;

      out[i] = (MYFLT)(((1.0f - mix) * convIn[0] + mix * wet) * fade);
    }

    return OK;
  }
};

// distnn_preload Spath — pre-warms the JSON cache at score time 0 so the first
// note-on doesn't pay the parse cost. Returns 0 (dummy i-rate output required
// by CPOF — Plugin<0,N> mis-computes inarg offsets).
struct DistNNPreload : csnd::Plugin<1, 1> {
  int init() {
    outargs[0] = 0;
    const char *path = inargs.str_data(0).data;
    if (g_json_cache.find(path) == g_json_cache.end()) {
      std::ifstream f(path);
      if (!f.is_open())
        return csound->init_error(
            std::string("distnn_preload: cannot open: ") + path);
      g_json_cache[path] = nlohmann::json::parse(f);
      csound->message(std::string("distnn_preload: cached ") + path + "\n");
    }
    return OK;
  }
};

void csnd::on_load(csnd::Csound *csound) {
  csnd::plugin<DistNN>(csound, "distnn", "a", "aSk", csnd::thread::ia);
  csnd::plugin<DistNNPreload>(csound, "distnn_preload", "i", "S",
                              csnd::thread::i);
}
