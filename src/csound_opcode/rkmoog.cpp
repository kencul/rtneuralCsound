#include <RKSimulationModel.h>
#include <vector>
#include <memory>
#include <modload.h>
#include <plugin.h>

// rkmoog aout, ain, kcutoff, kresonance
// Wraps RKSimulationMoog (RK4, 8x oversampled) as a Csound opcode for
// apples-to-apples CPU comparison with moognn.
struct RKMoog : csnd::Plugin<1, 3> {
  std::unique_ptr<RKSimulationMoog> filter;
  std::vector<float> scratch;
  float last_cutoff = -1.0f;
  float last_resonance = -1.0f;

  int init() {
    filter = std::make_unique<RKSimulationMoog>((float)sr());
    scratch.resize(insdshead->ksmps);  // nsmps not set until aperf(); ksmps is valid here
    last_cutoff    = -1.0f;
    last_resonance = -1.0f;
    return OK;
  }

  int deinit() {
    filter.reset();
    return OK;
  }

  int aperf() {
    if (!filter)
      return OK;

    MYFLT *out = outargs.data(0);
    MYFLT *in  = inargs.data(0);
    float cutoff    = (float)inargs[1];
    float resonance = (float)inargs[2];

    // SetCutoff/SetResonance are not free (SetCutoff multiplies by 2*pi),
    // so only call when the k-rate value actually changed.
    if (cutoff != last_cutoff) {
      filter->SetCutoff(cutoff);
      last_cutoff = cutoff;
    }
    if (resonance != last_resonance) {
      filter->SetResonance(resonance);
      last_resonance = resonance;
    }

    uint32_t n = nsmps - offset;
    for (uint32_t i = 0; i < n; i++)
      scratch[i] = (float)in[offset + i];

    filter->Process(scratch.data(), n);

    for (uint32_t i = 0; i < n; i++)
      out[offset + i] = (MYFLT)scratch[i];

    return OK;
  }
};

void csnd::on_load(csnd::Csound *csound) {
  csnd::plugin<RKMoog>(csound, "rkmoog", "a", "akk", csnd::thread::ia);
}
