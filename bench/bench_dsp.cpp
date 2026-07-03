// Benchmarks RKSimulationMoog CPU cost at varying polyphony levels.
// Cutoff mode "kr" (default) updates once per block, matching Csound k-rate dispatch.
// Cutoff mode "ar" updates per sample, matching how sweep_ref generated training targets.
// Output CSV schema is shared with the moognn opcode benchmark.

#include <RKSimulationModel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static constexpr float SR_DEFAULT     = 48000.0f;
static constexpr int   BLOCK_DEFAULT  = 64;
static constexpr float SECS_DEFAULT   = 10.0f;
static constexpr int   TRIALS_DEFAULT = 5;
static constexpr int   WARMUP_PASSES  = 2;

struct Config {
    float       sr       = SR_DEFAULT;
    int         block    = BLOCK_DEFAULT;
    float       seconds  = SECS_DEFAULT;
    int         trials   = TRIALS_DEFAULT;
    bool        ar_mode  = false;
    std::vector<int> voices = {1, 2, 4, 8, 16, 32, 64};
    std::string out      = "bench/results/dsp.csv";
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --block N           ksmps block size (default %d)\n"
        "  --sr N              sample rate (default %.0f)\n"
        "  --seconds N         audio duration per trial (default %.0f)\n"
        "  --trials N          timed trials per config (default %d)\n"
        "  --cutoff-mode kr|ar k-rate (default) or a-rate cutoff updates\n"
        "  --voices 1,2,4,...  comma-separated voice counts to sweep\n"
        "  --out path          output CSV (default bench/results/dsp.csv)\n",
        prog, BLOCK_DEFAULT, SR_DEFAULT, SECS_DEFAULT, TRIALS_DEFAULT);
}

static std::vector<int> parse_int_list(const char* s) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        out.push_back(std::stoi(tok));
    return out;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc)
            cfg.block = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--sr") && i + 1 < argc)
            cfg.sr = std::stof(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            cfg.seconds = std::stof(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc)
            cfg.trials = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--cutoff-mode") && i + 1 < argc)
            cfg.ar_mode = (strcmp(argv[++i], "ar") == 0);
        else if (!strcmp(argv[i], "--voices") && i + 1 < argc)
            cfg.voices = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            cfg.out = argv[++i];
        else {
            usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

// Log-sweep from 100 Hz to 20 kHz — exercises the full cutoff range.
static void build_kr_schedule(std::vector<float>& sched, int num_blocks) {
    constexpr float f0 = 100.0f, f1 = 20000.0f;
    sched.resize(num_blocks);
    for (int b = 0; b < num_blocks; b++) {
        float t = (float)b / (float)(num_blocks - 1);
        sched[b] = f0 * std::pow(f1 / f0, t);
    }
}

static void build_ar_schedule(std::vector<float>& sched, int total_samples) {
    constexpr float f0 = 100.0f, f1 = 20000.0f;
    sched.resize(total_samples);
    for (int s = 0; s < total_samples; s++) {
        float t = (float)s / (float)(total_samples - 1);
        sched[s] = f0 * std::pow(f1 / f0, t);
    }
}

static double run_pass(std::vector<RKSimulationMoog>& voices,
                       const std::vector<float>& input,
                       std::vector<float>& block_buf,
                       const std::vector<float>& kr_sched,
                       const std::vector<float>& ar_sched,
                       int block, int num_blocks, bool ar_mode,
                       volatile float& sink) {
    auto t0 = std::chrono::steady_clock::now();

    for (int b = 0; b < num_blocks; b++) {
        for (auto& v : voices) {
            // Each voice processes a fresh copy of the input block.
            std::copy(input.data() + b * block,
                      input.data() + b * block + block,
                      block_buf.begin());

            if (!ar_mode) {
                v.SetCutoff(kr_sched[b]);
                v.Process(block_buf.data(), (uint32_t)block);
            } else {
                for (int s = 0; s < block; s++) {
                    v.SetCutoff(ar_sched[b * block + s]);
                    v.Process(&block_buf[s], 1);
                }
            }

            // Single-sample accumulation into volatile sink prevents DCE.
            sink += block_buf[block / 2];
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    int total_samples = (int)(cfg.sr * cfg.seconds);
    int num_blocks    = total_samples / cfg.block;
    total_samples     = num_blocks * cfg.block;

    // 440 Hz sine input — deterministic, representative content.
    std::vector<float> input(total_samples);
    for (int i = 0; i < total_samples; i++)
        input[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * (float)i / cfg.sr);

    std::vector<float> kr_sched, ar_sched;
    build_kr_schedule(kr_sched, num_blocks);
    if (cfg.ar_mode)
        build_ar_schedule(ar_sched, total_samples);

    std::vector<float> block_buf(cfg.block);

    const char* mode_str = cfg.ar_mode ? "ar" : "kr";
    printf("bench_dsp  sr=%.0f  block=%d  seconds=%.1f  trials=%d  cutoff=%s\n",
           cfg.sr, cfg.block, cfg.seconds, cfg.trials, mode_str);

    bool write_header = !std::ifstream(cfg.out).good();
    FILE* csv = fopen(cfg.out.c_str(), "a");
    if (!csv) {
        fprintf(stderr, "Cannot open %s for writing\n", cfg.out.c_str());
        return 1;
    }
    if (write_header)
        fprintf(csv, "implementation,cutoff_mode,n_voices,block_size,sr,"
                     "total_samples,trial,wall_ms,rtf,cpu_pct,voices_at_realtime\n");

    volatile float sink = 0.0f;
    double audio_ms = (double)total_samples / (double)cfg.sr * 1000.0;

    for (int n_voices : cfg.voices) {
        std::vector<RKSimulationMoog> voices;
        voices.reserve(n_voices);
        for (int v = 0; v < n_voices; v++) {
            voices.emplace_back(cfg.sr);
            voices.back().SetResonance(0.5f);  // matches training data and bench_opcode.csd
        }

        printf("  voices=%-3d  warmup...", n_voices);
        fflush(stdout);

        for (int w = 0; w < WARMUP_PASSES; w++)
            run_pass(voices, input, block_buf, kr_sched, ar_sched,
                     cfg.block, num_blocks, cfg.ar_mode, sink);

        printf(" timing\n");

        std::vector<double> times;
        times.reserve(cfg.trials);

        for (int t = 0; t < cfg.trials; t++) {
            double ms = run_pass(voices, input, block_buf, kr_sched, ar_sched,
                                 cfg.block, num_blocks, cfg.ar_mode, sink);
            times.push_back(ms);
            printf("    trial %d  %.2f ms  rtf=%.4f\n", t + 1, ms, ms / audio_ms);
        }

        std::sort(times.begin(), times.end());
        if ((int)times.size() > 2) times.pop_back(); // drop slowest outlier

        double median_ms = times[times.size() / 2];
        double rtf       = median_ms / audio_ms;
        double cpu_pct   = rtf * 100.0;
        int    vrt       = (rtf > 0.0) ? (int)((double)n_voices / rtf) : 9999;

        printf("  -> median=%.2f ms  rtf=%.4f  cpu=%.1f%%  voices_at_realtime=%d\n\n",
               median_ms, rtf, cpu_pct, vrt);

        for (int t = 0; t < (int)times.size(); t++) {
            double wrtf = times[t] / audio_ms;
            fprintf(csv, "RKSimulation,%s,%d,%d,%.0f,%d,%d,%.4f,%.6f,%.4f,%d\n",
                    mode_str, n_voices, cfg.block, (double)cfg.sr, total_samples,
                    t + 1, times[t], wrtf, wrtf * 100.0, vrt);
        }
        fflush(csv);
    }

    fclose(csv);
    printf("(sink=%.6g)\n", (float)sink);
    return 0;
}
