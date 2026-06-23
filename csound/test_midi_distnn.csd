<CsoundSynthesizer>
<CsOptions>
--opcode-lib=build/bin/Release/distnn.dll
-o dac
-M a
-b 256
-B 4096
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 32
nchnls = 1
0dbfs = 1

massign 0, 1

instr 99
  iignore distnn_preload "models/dist_05_gru128/weights.json"
endin

instr 1
  ifreq  cpsmidi
  iamp   ampmidi 0.5

  kadsr  madsr 0.01, 0.1, 0.8, 0.3
  asine  poscil kadsr * iamp, ifreq

  ; CC 1 (mod wheel) controls wet/dry blend: 0 = dry, 127 = fully wet
  kmix   ctrl7 1, 14, 0, 1
  kmix   = portk:k(kmix, 0.01)

  aout   distnn asine, "models/dist_05_gru128/weights.json", kmix
         out asine

endin

</CsInstruments>
<CsScore>
i 99 0 0
f 0 3600
</CsScore>
</CsoundSynthesizer>
