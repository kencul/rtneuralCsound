<CsoundSynthesizer>
<CsOptions>
--opcode-lib=build/bin/Release/distnn.dll
-o distnn_test.wav
-W
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 64
nchnls = 1
0dbfs = 1

instr 1
  ain   diskin "audio/distortionBench.wav", 1
  kmix  = linseg:k(0, 15, 1)   ; ramp dry to wet, then hold wet
  aout  distnn ain, "models/dist_07_gru128_mrstft/weights.json", kmix
        out aout
endin
</CsInstruments>
<CsScore>
i 1 0 500
</CsScore>
</CsoundSynthesizer>
