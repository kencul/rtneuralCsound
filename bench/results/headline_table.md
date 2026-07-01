| Implementation | GRU units | CPU% / voice | Voices at realtime | Mean ESR (dB) |
|---|---|---|---|---|
| moognn 128u (deployed) | 128 | 17.0% | 5 | -45.9 |
| moognn 32u | 32 | 6.0% | 16 | -39.6 |
| moognn 64u | 64 | 8.7% | 11 | -42.3 |
| moognn 256u | 256 | 61.0% | 1 | -47.9 |
| distnn 128u (deployed) | 128 | 16.6% | 6 | — |
| moogladder (Csound DSP) | — | 0.7% | 132 | — |
| rkmoog (RK4 opcode) | — | 2.6% | 38 | — |
| RKSimulation (ODE ref.) | — | 2.3% | 42 | — |
