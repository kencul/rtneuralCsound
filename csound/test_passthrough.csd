<CsoundSynthesizer>
<CsOptions>
--opcode-lib=build/bin/Release/moognn.dll
-o moognn_test.wav
-W
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 64
nchnls = 1
0dbfs = 1

instr 1
  ain   diskin "audio/bench_mono.wav", 1
  klin = linseg:k(50, 2.5, 5000, 2.5, 50)
  aout  moognn128 ain, "models/19_moog_100-20k_128u_w256/weights.json", klin
        out aout
endin
</CsInstruments>
<CsScore>
i 1 0 5
</CsScore>
</CsoundSynthesizer>
