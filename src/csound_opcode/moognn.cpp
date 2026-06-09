// RTNeural and standard library must be included before Csound headers to avoid
// macro collisions from Csound's C API trampling C++ reserved words (e.g. in <chrono>).
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <RTNeural/RTNeural.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <plugin.h>
#include <modload.h>

static constexpr float FREQ_MIN = 20.0f;
static constexpr float FREQ_MAX = 20000.0f;
static constexpr float LOG_MIN  = 2.99573227f;  // log(20)
static constexpr float LOG_MAX  = 9.90348755f;  // log(20000)

static float normalizeKnob(float freqHz)
{
    float clamped = std::clamp(freqHz, FREQ_MIN, FREQ_MAX);
    return (std::log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN);
}

using ConvStage = RTNeural::ModelT<float, 1, 16,
    RTNeural::Conv1DT<float, 1, 16, 31, 1>>;

using RecurrentStage = RTNeural::ModelT<float, 17, 1,
    RTNeural::GRULayerT<float, 17, 64>,
    RTNeural::DenseT<float, 64, 1>>;

struct Model {
    ConvStage      conv;
    RecurrentStage rec;
};

// moognn aout, ain, Spath, kcutoff
struct MoogNN : csnd::Plugin<1, 3> {
    Model *model = nullptr;

    int init() {
        // Slot 1 is a string arg — Csound stores it as STRINGDAT* behind the MYFLT* pointer
        STRINGDAT *pathArg = (STRINGDAT*)inargs.begin()[1];
        const char *path = pathArg->data;

        std::ifstream f(path);
        if (!f.is_open())
            return csound->init_error(std::string("moognn: cannot open model: ") + path);

        nlohmann::json j;
        f >> j;

        model = new Model();
        RTNeural::torch_helpers::loadConv1D<float>(j, "conv.conv.", model->conv.get<0>());
        RTNeural::torch_helpers::loadGRU<float>   (j, "gru.",       model->rec.get<0>());
        RTNeural::torch_helpers::loadDense<float> (j, "dense.",     model->rec.get<1>());

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
        if (!model) return OK;

        MYFLT  *out  = &outargs[0];
        MYFLT  *in   = &inargs[0];
        float   knob = normalizeKnob((float)inargs[2]);

        float convIn[1], gruIn[17];

        for (uint32_t i = offset; i < nsmps; i++) {
            convIn[0] = (float)in[i];
            model->conv.forward(convIn);

            std::copy(model->conv.getOutputs(), model->conv.getOutputs() + 16, gruIn);
            gruIn[16] = knob;

            float result = model->rec.forward(gruIn);
            out[i] = (MYFLT)(result + convIn[0]);  // skip connection
        }

        return OK;
    }
};

void csnd::on_load(csnd::Csound *csound) {
    csnd::plugin<MoogNN>(csound, "moognn", "a", "aSk", csnd::thread::ia);
}
