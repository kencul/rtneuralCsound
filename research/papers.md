RELEVANT RESEARCH FOR MOOG FILTER VA AND DISTORTION EFFECT VA

---

## Project Background

This project develops a real-time neural network emulation of the Moog ladder filter, deployed as a Csound opcode, with distortion effect modeling as a planned second target. The primary deliverable is a working opcode suitable for a Csound conference paper.

**The emulation target** is the Moog ladder filter: a four-pole, 24dB/octave lowpass filter with a resonant feedback loop and nonlinear transistor ladder stages. It has a single real-time parameter — cutoff frequency (20Hz–20kHz on a log scale). Training data was generated using a C++ Moog ladder implementation run at 8x oversampling to eliminate Nyquist instability at high cutoffs, then downsampled to 48kHz.

**The model architecture** is `CausalConv1d(1→16, kernel=31) → concat cutoff knob → GRU(17→64) → Dense(64→1) + skip connection`. The Conv1d extracts temporal features from the audio; the knob is concatenated at every timestep as a 17th feature; the GRU carries the filter's recurrent state. The skip connection (raw input added to output) means the network learns the residual rather than the full signal. All layers are natively supported by RTNeural, enabling compile-time graph optimization and XSIMD-accelerated inference.

**Training** uses PyTorch with a causal warmup strategy: each window contains `warmup_size + window_size` samples, but loss is computed only on the `window_size` portion. This is a form of Truncated Backpropagation Through Time (TBPTT) that allows the GRU to reach a valid hidden state before gradients are registered. The best model (run 11, warmup=2048) achieves approximately −48dB ESR across the full 20Hz–20kHz range.

**The Csound opcode** (`moognn`) is a working CPOF plugin that loads model weights from a JSON file, runs per-sample RTNeural inference, and exposes a k-rate cutoff parameter. JSON caching eliminates file I/O on polyphonic note-on events. A 512-sample output fade-in partially masks a note-on transient click.

**Open problems:**

- *Per-note click*: The GRU cold-starts from zeros at every note-on. The model was trained with a 2048-sample warmup and has never learned to produce accurate output from sample zero. The architectural fix is `knob_to_h0` — a learned linear layer mapping the cutoff scalar to the GRU's initial hidden state — which was proven effective in an ablation study (contributing ~3dB accuracy, synergistic with LayerNorm for a combined ~10dB gain). A model with `knob_to_h0` restored needs to be retrained, and the opcode updated to seed `outs[]` before the first `forward()` call. The practical workaround until then is running the opcode as an always-on send effect rather than a per-note insert, which matches how a real Moog operates.
- *Dynamic cutoff behavior*: The model has only been trained and evaluated at static cutoffs. How it tracks a sweeping parameter mid-note is untested.
- *Distortion effects*: The pipeline (data generation → PyTorch training → RTNeural opcode) is designed to generalize. Distortion circuits (diode clippers, transistor saturation) are the intended next target, introducing more complex nonlinearities and potentially multiple parameters.

**Key architectural decisions and their rationale** are documented via a four-way ablation study across `knob_to_h0` and `LayerNorm` presence/absence, and a warmup-length series (2048 / 512 / 256 samples). Full results are in the DEVLOG.

---

## Architecture & Baselines

**1. Wright, Damskägg, Välimäki — "Real-time black-box modelling with recurrent neural networks"**
DAFx 2019
https://www.dafx.de/paper-archive/details.php?id=tieFMcaohHBxl_2yPyIGFA

Summary: Foundational paper showing LSTM and GRU can model nonlinear audio systems (tube amps, distortion pedals) in real time. Trained on recordings of guitar through target hardware. Achieved comparable accuracy to WaveNet at a fraction of the compute. LSTM slightly outperformed GRU.

Relevance: The GRU + Dense backbone used in this project is this paper's core finding. The additions here — CausalConv1d preprocessor, skip connection, knob_to_h0 — are extensions on top of this baseline. Essential citation as the architectural foundation.

---

**2. Simionato & Fasciani — "Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling"**
EURASIP Journal on Audio, Speech, and Music Processing, 2025
https://arxiv.org/abs/2405.04124

Summary: Compares LSTM, state-space models (SSMs), and linear recurrent units for VA audio effect modeling across distortion, EQ, saturation, and compression. LSTM outperformed alternatives for distortion and EQ. Encoder-decoder LSTM and SSM configurations excelled at compression and saturation. Evaluates both signal energy accuracy and transient reproduction.

Relevance: Directly compares architectures relevant to both current (filter) and future (distortion) work. The finding that LSTM/GRU performs best for distortion confirms the architecture choice. The encoder-decoder result for compression is relevant context for understanding when more complex architectures are justified. Also by the same authors as the SMC 2024 conditioning survey (entry 5).

---

**3. Wilczek, Wright, Välimäki, Habets — "Virtual Analog Modeling of Distortion Circuits Using Neural Ordinary Differential Equations"**
DAFx 2022
https://arxiv.org/abs/2205.01897

Summary: Models first-order and second-order diode clipper distortion circuits by learning the governing ODEs with neural networks. Achieves RNN-comparable accuracy with fewer parameters. The learned ODEs eliminate the need for oversampling and allow sample rate changes post-training without retraining.

Relevance: Directly relevant to future distortion work. The diode clipper is the canonical nonlinear distortion circuit. Neural ODEs are an alternative to RNNs that model the continuous-time nature of analog circuits more naturally, which is physically principled for circuits defined by differential equations. The sample rate flexibility is a practical deployment advantage over the current approach, where the model is fixed to 48kHz.

---

**4. Carson, Wright, Chowdhury, Välimäki, Bilbao — "Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing"**
DAFx 2024
https://arxiv.org/abs/2406.06293

Summary: Addresses RNNs trained at one sample rate producing unreliable results at others. Proposes delay-based and Lagrange interpolation methods to achieve approximate sample rate independence. Demonstrates high-fidelity integer oversampling and non-integer rate conversion with reduced aliasing.

Relevance: This model is trained at 48kHz. If the Csound opcode is used in a project running at 44.1kHz, the weights would behave incorrectly without modification. The delay-based method described here is compatible with RTNeural's architecture and provides a path to making the opcode robust across different Csound sample rate configurations. Relevant for both filter and future distortion deployment.

---

## Parameter Conditioning

**5. Wenke & Fleming — "Contextual Recurrent Neural Networks"**
arXiv 2019
https://arxiv.org/abs/1902.03455

Summary: Proposes conditioning the RNN initial hidden state h0 on contextual input information rather than defaulting to zeros. The h0 is parameterized and trained end-to-end. Shows performance improvements on associative retrieval tasks.

Relevance: Closest academic precedent for knob_to_h0. The principle is identical — h0 derived from a learned function of a conditioning input rather than cold-starting from zero. The contribution here is applying this to parametric virtual analog modeling with a scalar control parameter, and demonstrating empirically via ablation that it is load-bearing for low-frequency accuracy. Cite this as the general principle that knob_to_h0 instantiates.

---

**6. Yeh, Hsiao, Yang — "Hyper Recurrent Neural Network: Condition Mechanisms for Black-box Audio Effect Modeling"**
DAFx 2024
https://arxiv.org/abs/2408.04829

Summary: Identifies input concatenation of control parameters as insufficient for conditioning RNNs in audio effect modeling. Proposes three alternatives: FiLM-RNN (scale/shift from hidden state), StaticHyper-RNN (hypernetwork generates fixed weights from parameters), and DynamicHyper-RNN (weights generated per timestep). Outperforms concatenation-based baselines across multiple effects.

Relevance: Directly validates the motivation behind knob_to_h0 — naive knob concatenation to audio input is suboptimal. Their conclusion (better conditioning mechanisms are needed) is the same problem solved differently here. HyperRNN conditions during inference via weight generation; knob_to_h0 conditions at initialization via h0 seeding. Positions this work within the current state of the field and is a strong contrast citation. Applies to future distortion work with multiple control parameters.

---

**7. Comunità, Steinmetz, Phan, Reiss — "Modelling black-box audio effects with time-varying feature modulation"**
ICASSP 2023
https://arxiv.org/abs/2211.00497

Summary: Addresses long-horizon audio effects (compression, fuzz) by integrating time-varying FiLM into temporal convolutional backbones. FiLM modulates intermediate feature activations during inference rather than only at the input. Improves modeling of effects with long temporal dependencies.

Relevance: FiLM is the dominant conditioning approach in the literature. Useful contrast to knob_to_h0 — FiLM conditions features at every timestep during inference; knob_to_h0 conditions the recurrent state once at initialization. For a Moog LPF with short memory and a single parameter, the lighter h0 approach is more appropriate. For distortion with multiple interacting controls (gain, tone, presence), FiLM may be worth revisiting.

---

**8. Simionato & Fasciani — "Deep Learning Conditioned Modeling of Optical Compression"**
DAFx 2022
https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_6.pdf

Summary: First application of an encoder-decoder architecture to conditioned black-box modeling of optical compression. Compares feedforward DNN, LSTM, and LSTM with encoder-decoder conditioning across two continuous parameters (ratio and threshold). The encoder processes a reference input-output segment to extract a latent state that seeds the LSTM decoder's initial hidden state, bypassing the need to supply parameter values explicitly. Encoder-decoder outperforms both baselines, particularly in reproducing the attack transient phase.

Relevance: Establishes the encoder-decoder paradigm for seeding an RNN hidden state from context in audio effect modeling — the same mechanism as knob_to_h0, but deriving h0 from audio context rather than known parameter values. The contrast is useful for framing knob_to_h0: when the parameter value is known (as with a Moog cutoff dial), seeding directly from the scalar is simpler and sufficient; the encoder approach is needed only when parameters must be inferred from audio. This is part of a progression by the same authors — DAFx 2022 → DAFx 2023 (entry 9) → EURASIP 2025 (entry 2) — establishing the encoder-decoder method for compression modeling.

---

**9. Simionato & Fasciani — "Fully Conditioned and Low-Latency Black-Box Modeling of Analog Compression"**
DAFx 2023
https://www.dafx.de/paper-archive/2023/DAFx23_paper_10.pdf

Summary: Extends the 2022 encoder-decoder conditioning architecture to four parameters — ratio, threshold, attack time, and release time — and evaluates it across seven optical and VCA compressor devices. Attack and release times are temporal parameters that affect the trajectory of the response over time, making this a time-varying conditioning problem. Proposes a low-latency variant suitable for real-time deployment. Encoder-decoder continues to outperform feedforward and LSTM baselines.

Relevance: Directly relevant to the dynamic parameter tracking problem in this project. Attack and release times behave analogously to a sweeping cutoff — they change how the model transitions between states over time, not just the steady-state behavior. The architecture handling these time-variant parameters (encoder seeding + per-sample knob injection) is the most complete prior example of multi-parameter conditioning in audio effect modeling before this project. The low-latency design is directly applicable to real-time Csound deployment. Shares authorship with entry 2 (EURASIP 2025 comparative study) and entry 8 (DAFx 2022).

---

**11. Simionato & Fasciani — "Conditioning Methods for Neural Audio Effects"**
SMC 2024
https://smcnetwork.org/smc2024/papers/SMC2024_paper_id83.pdf

Summary: Experimental comparison of conditioning mechanisms for neural audio effect models, using an S4D (diagonal state-space) backbone applied to overdrive and compression. Three conditioning approaches are compared against a concatenation baseline: Gated Activation (GA), FiLM-GLU (affine transformation + gated linear unit with softsign), and FiLM-GCU (same but with convolutional gated unit). Each is also tested at three placement positions: before the S4D layer (pre), after (post), and both (pre-post). Nonlinear FiLM transformations (cubic, quintic) are additionally tested. Training data uses 3 equally spaced values per parameter across 9 parameter combinations per effect, at 48kHz.

Key findings:
- FiLM outperforms GA and concatenation baseline in all cases
- Post-placement (conditioning applied after the state layer) consistently beats pre or pre-post
- For compression: FiLM-GLU (linear FC in gated unit) works best
- For overdrive: FiLM-GCU (convolutional gated unit) works best
- Nonlinear FiLM transformations (3rd order) improve overdrive accuracy; no benefit for compression
- Input concatenation is consistently the worst conditioning method

Relevance: Directly relevant to both the current Moog filter work and future distortion modeling. The finding that input concatenation performs worst validates the motivation for knob_to_h0 as an alternative — both papers independently conclude that passing the control scalar directly into the audio stream is insufficient. The post-placement result (condition after the recurrent layer, not before) is a practical guideline for future FiLM experiments: if FiLM is added to the distortion pipeline, place it after the GRU output, not before. The overdrive-vs-compression split (convolutional vs linear gated unit) is useful for architecture selection when moving to multi-parameter distortion targets. The 3-value-per-parameter grid sampling is a minimum viable dataset density guideline. Note: this paper uses S4D as the backbone rather than GRU — placement findings should generalize but are not directly measured on GRU architectures.

---

## Training Methodology

**12. Bourdin, Legrand, Roche et al. — "Empirical Results for Adjusting Truncated Backpropagation Through Time while Training Neural Audio Effects"**
DAFx 2025 (arXiv 2512.07393); extended from EA 2024 (Bourdin et al., "Tackling Long-Range Dependencies in Dynamic Range Compression Modeling via Deep Learning," Bordeaux)
https://arxiv.org/abs/2512.07393

Summary: Systematically evaluates TBPTT hyperparameters — sequence number, batch size, and sequence length — for training neural audio effect models targeting dynamic range compression. Uses the SPTMod convolutional-recurrent architecture. Demonstrates that tuning these parameters improves accuracy and training stability while reducing compute. The EA 2024 conference paper is the earlier presentation where SPTMod and the State Prediction Network (SPN) concept were first introduced; the DAFx 2025 arXiv version adds the TBPTT hyperparameter analysis.

Relevance: The warmup system in this project is a specific form of TBPTT — gradients are only computed after the first warmup_size samples, which is equivalent to treating the warmup as the truncation window. The findings give theoretical grounding for the warmup length experiments (runs 11/12/13) and validate the warmup approach as established practice. The SPTMod architecture from this group's prior work is the more complex architecture discussed in research notes.

---

**13. Steinmetz & Reiss — "Efficient neural networks for real-time modeling of analog dynamic range compression"**
AES Convention 2022
https://arxiv.org/abs/2102.06200

Summary: Proposes optimized temporal convolutional networks (TCNs) with rapidly growing dilations to model the LA-2A compressor in real time on CPU. Achieves state-of-the-art accuracy with only ten minutes of training data. Sparse dilated convolutions expand the receptive field while keeping inference cost low.

Relevance: TCNs with growing dilations are the main alternative to GRU for audio effects with long memory. For distortion effects (short memory, sample-by-sample nonlinearity), GRU remains more appropriate — but this paper establishes the TCN baseline and the receptive-field tradeoffs. Understanding when to use TCN vs GRU is relevant for future multi-effect work.

---

## Real-Time Deployment

**14. Chowdhury — "RTNeural: Fast Neural Inferencing for Real-Time Systems"**
arXiv 2021
https://arxiv.org/abs/2106.03037

Summary: Introduces RTNeural, a C++ library for real-time neural network inference under hard real-time constraints. Describes design principles emphasizing speed, small binary size, and flexibility. Benchmarks against competing inference libraries for audio-rate processing.

Relevance: RTNeural is the inference engine powering the Csound opcode. Cite this whenever the opcode or inference pipeline is described. The compile-time templated API, XSIMD intrinsics, and stateful GRU behavior described in this paper are the specific features relied upon in this project.

---

## Dynamic Parameter Tracking

**15. Bourdin, Legrand, Roche — "Time-Varying Audio Effect Modeling by End-to-End Adversarial Training"**
JAES 2025 (submitted)
https://arxiv.org/abs/2512.15313

Summary: Presents a GAN-based approach for modeling time-varying audio effects from input-output recordings. Combines a convolutional-recurrent architecture with a two-stage training strategy (adversarial phase followed by supervised fine-tuning). Uses a State Prediction Network to synchronize model state with targets. Introduces a chirp-train metric for evaluating modulation accuracy. Demonstrated on vintage phasers.

Relevance: Addresses the dynamic cutoff behavior problem flagged in the devlog as high priority before Csound deployment. The State Prediction Network (SPN) used here for state synchronization is conceptually related to knob_to_h0 but more complex — it predicts state from audio context rather than just the parameter. The chirp-train evaluation metric for assessing modulation accuracy is directly applicable to testing how well the current model tracks a sweeping cutoff. Relevant for both the dynamic cutoff evaluation experiment and future distortion with time-varying parameters.

---

## Exposure Bias & Training Stability

**16. Peussa, Damskägg, Sherson, Mimilakis, Juvela, Gotsopoulos, Välimäki — "Exposure Bias and State Matching in Recurrent Neural Network Virtual Analog Models"**
DAFx 2021
https://aaltodoc.aalto.fi/items/53cf76e8-595b-405d-aaed-d8a3e0ea8bba

Summary: Identifies exposure bias as a core failure mode in free-running RNN audio models — where error accumulates because the network was trained on ground-truth hidden states but must run on its own imperfect states at inference time. Proposes a state-matching mechanism for GRU networks and demonstrates that truncated backpropagation through time substantially reduces vulnerability to this, particularly for circuits with external modulation.

Relevance: This paper formally names and analyzes the exact problem the warmup strategy in this project is designed to combat. The 2048-sample causal warmup is precisely a form of TBPTT that avoids feeding the GRU artificially perfect states during training. The state-matching mechanism described is also a direct precedent for knob_to_h0 — both are approaches to seeding the GRU into a valid state before loss is measured. Essential theoretical grounding for the training methodology section of the paper.

---

## Anti-Aliasing for Distortion Deployment

**17. Köper & Holters — "Antialiased State Trajectory Neural Networks for Virtual Analog Modeling"**
DAFx 2023
https://www.dafx.de/paper-archive/2023/DAFx23_paper_53.pdf

Summary: Integrates Antiderivative Antialiasing (ADAA) directly into State Trajectory Networks for neural virtual analog modeling. ADAA analytically suppresses aliasing by computing a discrete difference quotient of the antiderivative of the nonlinear activation function rather than evaluating it directly. Eliminates high-frequency foldback artifacts introduced by nonlinear activations without requiring oversampling.

Relevance: ADAA is the principled mathematical solution to aliasing in distortion networks — the alternative to brute-force oversampling. For a Moog LPF the aliasing problem is less severe (handled at data generation time via 8x oversampling of the C++ reference), but for distortion circuits driven into hard clipping, the nonlinear activations inside the neural network itself generate harmonics above Nyquist. This paper shows how to suppress those analytically inside the network architecture.

---

**18. Carson, Wright, Bilbao — "Anti-aliasing of neural distortion effects via model fine tuning"**
DAFx 2025 / arXiv 2025
https://arxiv.org/abs/2505.11375

Summary: Presents a teacher-student fine-tuning approach where a pre-trained baseline model (teacher) guides a student copy trained against aliasing-free targets generated by removing non-harmonic spectral components. Significantly reduces aliasing in both LSTM and TCN architectures, outperforming 2x oversampling in most cases. LSTM models achieve the best balance between aliasing reduction and similarity to the analog reference.

Relevance: A practical, data-driven alternative to ADAA that requires no architectural changes — you just fine-tune an existing trained model against alias-free targets. For the distortion pipeline, this means the same Conv1D → GRU → Dense architecture can be retrained to suppress aliasing without modifying the C++ inference code or the RTNeural integration. Particularly relevant because it confirms LSTM/GRU outperforms TCN for this task.

---

**19. Esqueda & Murai — "Antialiased Black-Box Modeling of Audio Distortion Circuits using Real Linear Recurrent Units"**
DAFx 2025
https://dafx25.dii.univpm.it/wp-content/uploads/2025/09/DAFx25_paper_61.pdf

Summary: Proposes interleaving real-valued Linear Recurrent Units (LRUs) with static nonlinear activations for black-box distortion modeling. LRUs separate linear memory from nonlinear waveshaping, allowing the nonlinear components to be antialiased via ADAA independently. Applied to second-order diode clippers and overdrive pedals. Achieves high accuracy with low parameter count and parallelizable training.

Relevance: Represents an alternative to GRU for distortion specifically — LRUs are parallelizable (unlike GRUs, which are inherently sequential) and pair cleanly with ADAA because the nonlinearities are isolated from the recurrent components. If future distortion work demands aliasing-free inference without teacher-student fine-tuning, this architecture is the most principled option. Not compatible with RTNeural's native layer set, so would require custom inference.

---

**20. Sato & Smith — "Aliasing Reduction in Neural Amp Modeling by Smoothing Activations"**
DAFx 2025
https://arxiv.org/abs/2505.04082

Summary: Addresses aliasing in neural guitar amp emulation by replacing standard activations (tanh, ReLU) with smoother variants (custom tanh, Snake). Introduces the Aliasing-to-Signal Ratio (ASR) metric to quantify aliasing independently of modeling accuracy. Demonstrates that smoother activation curves reduce aliasing while maintaining accuracy.

Relevance: The simplest possible approach to aliasing reduction — swap activation functions, no architectural changes. For the current Moog LPF model, tanh was already removed because the filter is linear. For future distortion models where nonlinear activations are load-bearing, this paper suggests that activation choice directly affects aliasing behavior and provides a metric (ASR) to measure it objectively.

---

## Data

**21. Comunità et al. — ToneTwist AFx Dataset**
Dataset / GitHub
https://github.com/mcomunita/tonetwist-afx-dataset

Summary: The largest open-source collection of dry/wet signal pairs for nonlinear audio effect modeling — 40+ analog and digital devices across tube amplifiers, compressors, overdrive circuits, fuzz pedals, and preamplifiers. Seven audio source types including guitar, bass, and test signals. Devices categorized by parametric nature (fixed vs. variable controls) and technology.

Relevance: The standard benchmark dataset for distortion and saturation modeling. When moving from the Moog filter to distortion circuits, this dataset eliminates the need to record or simulate training data from scratch for common target devices. The parametric device subset (tube amps with multiple controls) is directly relevant to multi-parameter conditioning experiments. The fixed-parameter devices (overdrive pedals, fuzz) are the simplest distortion targets to start with.

---

## Gray-Box & Differentiable Approaches

**22. (Authors TBC) — "A Differentiable Digital Moog Filter For Machine Learning Applications"**
DAFx 2023
https://www.dafx.de/paper-archive/2023/DAFx23_paper_14.pdf

Summary: Derives analytical backpropagation expressions for a simplified digital Moog ladder filter, enabling it to function as a differentiable layer inside a machine learning pipeline. Demonstrates uses in adaptive filtering and as a trainable component in deep learning systems. Generalizes the differentiation method to multi-stage ladder topologies.

Relevance: A gray-box alternative to the fully black-box RNN approach used in this project. By embedding a differentiable Moog model as a layer, a network can be constrained to produce physically valid filter behavior while learning residual corrections from data. Not the approach taken here, but directly relevant to the Moog target and useful context for discussing the white-box vs. black-box tradeoff in the paper.

---

## Paper Relationships

- **Wright 2019** is the architectural baseline everything builds from.
- **Wenke & Fleming 2019** is the academic origin of the knob_to_h0 principle.
- **Peussa 2021** formally defines the exposure bias problem that the warmup strategy addresses; state-matching is a direct precedent for knob_to_h0.
- **Simionato 2022 (DAFx, entry 8)** and **Simionato 2023 (DAFx, entry 9)** form the encoder-decoder progression: 2022 establishes seeding LSTM h0 from audio context; 2023 extends to four parameters including time-variant ones. Both are the same mechanism as knob_to_h0 applied to a blind (audio-derived) conditioning problem rather than an explicit parameter. Together with entries 2 and 11, they trace the full Simionato & Fasciani line of work on conditioning.
- **HyperRNN (Yeh 2024)** and **Comunità 2023** are contemporary alternatives to the same conditioning problem, reached via different mechanisms.
- **Simionato 2025 (EURASIP, entry 2)** and **Simionato 2024 (SMC, entry 11)** together provide both an architectural comparison and a conditioning taxonomy from the same group.
- **Bourdin 2025 (DAFx, entry 12)** grounds the warmup/TBPTT training approach in formal methodology; the EA 2024 conference paper is the earlier presentation where SPTMod/SPN was introduced; **Peussa 2021 (entry 16)** provides the earlier theoretical justification for exposure bias.
- **Bourdin 2025 (JAES, entry 15)** addresses dynamic parameter tracking with an SPN variant.
- **Chowdhury 2021 (entry 14)** is the citation for RTNeural itself.
- **Wilczek 2022 (entry 3)** and **Carson 2024 (entry 4)** are forward references for distortion and cross-sample-rate deployment.
- **Anti-aliasing cluster (Köper 2023/entry 17, Carson 2025/entry 18, Esqueda 2025/entry 19, Sato 2025/entry 20)** covers the aliasing problem for distortion from four angles: ADAA integration, teacher-student fine-tuning, LRU architecture, and activation smoothing.
- **ToneTwist AFx (entry 21)** is the training data source for future distortion work.
- **Differentiable Moog (entry 22)** is the gray-box contrast to the black-box approach taken here.


https://nva.sikt.no/registration/0198cc80a594-816602e1-c533-4d70-a45e-a0b8606fa3d0
Max msp program for collecting data with dynamic params

https://www.dafx.de/paper-archive/2023/DAFx23_paper_14.pdf
KORG paper for realLRU