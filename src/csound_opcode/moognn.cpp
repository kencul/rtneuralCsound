#include <plugin.h>
#include <modload.h>
#include <algorithm>

// moognn aout, ain, kcutoff
// Passthrough stub — audio flows through unchanged, cutoff ignored.
// Replace aperf() body with RTNeural inference once build pipeline is confirmed.
struct MoogNN : csnd::Plugin<1, 2> {
    int init() {
        return OK;
    }

    int aperf() {
        MYFLT *out = &outargs[0];
        MYFLT *in  = &inargs[0];
        // sa_offset() already zeroed the offset/early regions of out
        std::copy(in + offset, in + nsmps, out + offset);
        return OK;
    }
};

void csnd::on_load(csnd::Csound *csound) {
    csnd::plugin<MoogNN>(csound, "moognn", "a", "ak", csnd::thread::ia);
}
