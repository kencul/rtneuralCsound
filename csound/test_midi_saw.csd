<CsoundSynthesizer>
<CsOptions>
--opcode-lib=build/bin/Release/moognn.dll
-o dac
-M a
-b 256
-B 4096
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 256
nchnls = 1
0dbfs = 1

gisaw    ftgen 3, 0, 1024, 7, 1, 512, 1, 0, -1, 512, -1

massign 0, 1

instr 99
  iignore moognn_preload "ref/14_moog_20-20k_64u_k2h0_LN/weights.json"
endin

instr 1
  ifreq  cpsmidi
  iamp   ampmidi 0.5

  kadsr madsr 0.01, 0.1, 0.8, 0.3

  asaw   poscil kadsr * iamp, ifreq, gisaw      ; sawtooth

  kcc      ctrl7 1, 14, 0, 1
  kcc = portk:k(kcc, 0.01)
  kcutoff  = 20 * pow(1000, kcc)

  aout   moognn asaw, "ref/14_moog_20-20k_64u_k2h0_LN/weights.json", kcutoff
  out aout
endin

</CsInstruments>
<CsScore>
i 99 0 0
f 0 3600
</CsScore>
</CsoundSynthesizer>
