| Implementation | GRU units | CPU% / voice | Voices at realtime | Mean ESR (dB) |
|---|---|---|---|---|
| moognn 128u (deployed) | 128 | 17.2% | 5 | -45.9 |
| moognn 32u | 32 | 5.9% | 16 | -39.6 |
| moognn 64u | 64 | 8.6% | 11 | -42.3 |
| moognn 256u | 256 | 60.3% | 1 | -47.9 |
| distnn 128u (deployed) | 128 | 16.5% | 6 | — |
| moogladder (Csound DSP) | — | 0.7% | 141 | — |
| rkmoog (RK4 opcode) | — | 2.6% | 39 | — |
| RKSimulation (ODE ref.) | — | 2.3% | 42 | — |
