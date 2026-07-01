<CsoundSynthesizer>
<CsOptions>
--opcode-lib=build/bin/Release/moognn.dll
--opcode-lib=build/bin/Release/distnn.dll
--opcode-lib=build/bin/Release/rkmoog.dll
-d
-m0
</CsOptions>
<CsInstruments>
sr      = 48000
ksmps   = 64
nchnls  = 1
0dbfs   = 1

; ----------------------------------------------------------------------
; Macros (set via --omacro:NAME=VALUE on the command line):
;   INSTR    1=moognn, 2=distnn, 3=moogladder, 4=rkmoog
;   OPCODE   moognn opcode name: moognn32 / moognn64 / moognn128 / moognn256
;            (must match the GRU hidden size in WEIGHTS; ignored for INSTR != 1)
;   WEIGHTS  path to weights JSON (ignored by moogladder and rkmoog)
;   VOICES   polyphony count
;   DUR      render duration in seconds
; Defaults below let the .csd run standalone for sanity checks.
; ----------------------------------------------------------------------

#ifndef INSTR
#define INSTR #3#
#end

#ifndef OPCODE
#define OPCODE #moognn128#
#end

#ifndef WEIGHTS
#define WEIGHTS #models/19_moog_100-20k_128u_w256/weights.json#
#end

#ifndef VOICES
#define VOICES #1#
#end

#ifndef DUR
#define DUR #10#
#end

; Pre-warm JSON cache so the first voice doesn't pay parse cost.
instr 99
  if $INSTR == 1 then
    iignore moognn_preload "$WEIGHTS"
  elseif $INSTR == 2 then
    iignore distnn_preload "$WEIGHTS"
  endif
  turnoff
endin

; Spawn $VOICES instances of instr $INSTR simultaneously at t=0.
; Stays alive for $DUR seconds so the performance doesn't terminate before
; the scheduled voices finish, then self-terminates cleanly.
instr 100
  iN = $VOICES
  iI = 0
  while iI < iN do
    schedule $INSTR, 0, $DUR
    iI = iI + 1
  od
  ; Terminate performance once voices finish. Latch prevents repeated "e" events.
  kTime   timeinsts
  kEnded  init 0
  if kTime >= $DUR && kEnded == 0 then
    event "e", 0, 0
    kEnded = 1
  endif
endin

; Voice 1: moognn (size selected via $OPCODE macro). Log cutoff sweep
; 100Hz -> 20kHz matches bench_dsp.cpp.
instr 1
  aIn     poscil  0.5, 440
  kCut    expon   100, p3, 20000
  aOut    $OPCODE aIn, "$WEIGHTS", kCut
          out     aOut
endin

; Voice 2: distnn. Mix ramp 0 -> 1 exercises wet path under load.
instr 2
  aIn     poscil  0.5, 440
  kMix    line    0, p3, 1
  aOut    distnn  aIn, "$WEIGHTS", kMix
          out     aOut
endin

; Voice 3: moogladder. Native Csound DSP baseline.
instr 3
  aIn     poscil      0.5, 440
  kCut    expon       100, p3, 20000
  kRes    =           0.5
  aOut    moogladder  aIn, kCut, kRes
          out         aOut
endin

; Voice 4: rkmoog. RK4 + 8x oversampled C++ reference; apples-to-apples
; comparison with moognn. Same sweep and resonance as training data.
instr 4
  aIn     poscil  0.5, 440
  kCut    expon   100, p3, 20000
  kRes    =       0.5
  aOut    rkmoog  aIn, kCut, kRes
          out     aOut
endin

</CsInstruments>
<CsScore>
i 99 0 0
i 100 0 3600
e
</CsScore>
</CsoundSynthesizer>
