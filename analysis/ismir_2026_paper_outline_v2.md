# ISMIR 2026 Paper Outline: JazzFlow-RT (REVISED)

**Title**: JazzFlow-RT: Investigating Real-Time Jazz Improvisation through Hybrid Architecture and Knowledge Distillation

**Authors**: [Your Name], [Collaborators]

**Venue**: ISMIR 2026 (International Society for Music Information Retrieval)

**Submission Deadline**: April 2026

**Paper Type**: Full paper (6-8 pages + references)

**Status**: Research Proposal (Results Pending)

---

## ⚠️ IMPORTANT NOTE

This outline presents **hypothesized results** based on component capabilities. Actual results may differ significantly. The paper will report **actual measured results**, not predictions.

---

## Abstract (250 words) - CONSERVATIVE VERSION

**Draft**:

Jazz improvisation generation faces a fundamental trade-off between quality and latency. State-of-the-art models like ImprovNet achieve high jazz style recognition (79%) but require 8 seconds for 512 tokens, while real-time models like Magenta RealTime generate in <50ms but with poor jazz quality (55%). We investigate whether this trade-off is fundamental or can be overcome through hybrid architecture design and knowledge distillation.

We propose **JazzFlow-RT**, a two-model system combining Music Informer's ProbSparse Attention, ImprovNet's corruption-refinement learning, and streaming transformers. Our teacher model hypothesizes that ProbSparse Attention can accelerate encoding without degrading jazz quality. Our student model tests whether jazz "feel" can be distilled into a lightweight model for real-time performance.

**Critical research questions validated through experiments:**
1. Does ProbSparse Attention preserve jazz quality, or does it miss important syncopation patterns?
2. Are ProbSparse and corruption-refinement compatible, or do they conflict?
3. What percentage of jazz quality transfers via distillation in music (vs 97% in NLP)?
4. Can <100ms end-to-end latency be achieved with realistic system overhead?

**Results** (to be measured): Our teacher model achieves [X%] jazz recognition (vs ImprovNet's 79%) at [Y]s latency. Our student model achieves [Z%] recognition at [W]ms end-to-end latency. Ablation studies reveal [findings about ProbSparse compatibility]. We provide comprehensive analysis of when and why each component succeeds or fails, advancing understanding of efficient music generation.

**Keywords**: jazz generation, real-time music generation, transformer efficiency, knowledge distillation, corruption-refinement learning, empirical evaluation

---

## 1. Introduction (1 page)

### 1.1 Motivation & Research Gap

**The Quality-Latency Trade-Off**:

Current jazz generation models fall into two categories:
- **High-quality, slow**: ImprovNet (79% recognition, 8s latency), Music Transformer (64%, 5s)
- **Low-quality, fast**: Magenta RealTime (55% recognition, 50ms latency)

**Research Questions**:
1. Is this trade-off fundamental, or can it be overcome?
2. Can we combine techniques from time-series forecasting (ProbSparse), music generation (corruption-refinement), and real-time systems (streaming) successfully?

### 1.2 Contributions

**Primary Contribution**:
✅ **Empirical investigation** of combining three techniques:
- ProbSparse Attention (from time-series forecasting) for music
- Corruption-refinement (from ImprovNet) with efficient attention
- Knowledge distillation for musical style transfer

**Secondary Contributions**:
✅ **Comprehensive ablation studies** revealing:
- When ProbSparse Attention works/fails for music
- Compatibility of ProbSparse + corruption-refinement
- Importance ranking of 9 corruption functions
- Distillation retention rate for musical "feel"

✅ **Fair baseline comparison**:
- All baselines fine-tuned on same jazz dataset
- Statistical significance testing
- Effect size reporting

✅ **Open-source release**:
- Complete implementation
- Trained models
- Evaluation code
- Negative results documented

**Tertiary Contribution** (if successful):
⚠️ First system achieving >65% jazz recognition at <100ms latency

### 1.3 Hypotheses & Validation

**H1**: ProbSparse Attention preserves jazz quality
- **Test**: Ablation study (ProbSparse vs Standard + Corruption)
- **Success criterion**: <5% quality degradation
- **Risk**: May miss syncopation patterns → >5% degradation

**H2**: ProbSparse + Corruption are compatible
- **Test**: Dedicated compatibility experiment
- **Success criterion**: No architectural conflict
- **Risk**: ProbSparse may skip corrupted positions

**H3**: Jazz quality distills at 75-85% retention
- **Test**: Teacher→Student knowledge transfer
- **Success criterion**: 75-85% of teacher quality
- **Risk**: Musical "feel" may not distill well → 60-75%

**H4**: <100ms end-to-end latency achievable
- **Test**: Realistic pipeline measurement
- **Success criterion**: p95 < 100ms
- **Risk**: System overhead may dominate → 100-150ms

---

## 2. Related Work (1-1.5 pages)

[Same structure as original, but add section:]

### 2.5 Research Gap & Open Questions

**Unanswered Questions**:
1. **ProbSparse for Music**: Informer designed for smooth time-series (electricity, weather). Does it work for discrete, rhythmic music with sudden changes (jazz syncopation)?

2. **Distillation for Music**: DistilBERT achieves 97% retention in NLP. What about music where subtle timing (ms-level) and dynamics matter?

3. **Hybrid Architecture**: No prior work combines ProbSparse + Corruption-refinement. Are they compatible?

4. **Real-Time Jazz**: No existing work >65% recognition at <100ms latency. Is it feasible?

**Our Contribution**: Empirical answers to these questions through rigorous experimentation.

---

## 3. Methods (2.5-3 pages)

### 3.1 Problem Formulation

**Input**: Source melody S = (s₁, ..., sₙ)
**Output**: Jazz improvisation I = (i₁, ..., iₘ)

**Objectives**:
1. **Quality**: I should exhibit jazz style (>65% recognition target)
2. **Efficiency** (Teacher): Faster encoding than baseline
3. **Latency** (Student): <100ms end-to-end (p95)

### 3.2 Teacher Model Architecture

[Same as original but add:]

**Design Rationale & Risks**:

**Why ProbSparse in Encoder?**
- Pro: 21.73% faster encoding (proven in Informer paper)
- Con: Designed for smooth time-series, not discrete music
- **Risk**: May miss important musical patterns → validate empirically

**Why Deep Decoder (12 layers)?**
- Pro: More expressive capacity for complex jazz patterns
- Con: More parameters (80M) → overfitting risk with 200h dataset
- **Mitigation**: Strong regularization, data augmentation

**Why Corruption-Refinement Training?**
- Pro: ImprovNet achieves 79% with this approach
- Con: May conflict with ProbSparse (both affect attention patterns)
- **Risk**: Need compatibility test (Section 5.3)

### 3.3 Critical Compatibility Test

**Research Question**: Are ProbSparse and Corruption compatible?

**Potential Conflict**:
```python
# ProbSparse selects top-u "informative" queries
# But corruption creates MASK tokens everywhere
# Will ProbSparse skip masked positions?

Example:
Corrupted: [100, MASK, MASK, 900, ...]
ProbSparse computes M-score for each query
IF M-score(MASK queries) is low:
    → ProbSparse skips them
    → Cannot restore masked tokens
    → Training fails
```

**Experiment Design**:
```python
# Test 1: Standard Attention + Corruption (baseline)
model_standard = Encoder(use_prob_sparse=False) + Decoder()
result_standard = train_and_eval(model_standard, corruption_data)

# Test 2: ProbSparse + Corruption (proposed)
model_probsparse = Encoder(use_prob_sparse=True) + Decoder()
result_probsparse = train_and_eval(model_probsparse, corruption_data)

# Analysis
degradation = result_standard - result_probsparse

IF degradation < 5%:
    compatible = True  # Proceed with hybrid
ELIF degradation > 10%:
    compatible = False  # Use Standard Attention
ELSE:
    compatible = Marginal  # Try hybrid (some layers ProbSparse)
```

**This test is CRITICAL - conducted in Month 20, Week 3-4**

### 3.4 Training Strategy

**Challenge**: 200h PiJAMA dataset is small for 115M parameters

**Solution: Aggressive Data Augmentation**
```python
Augmentations:
- Pitch transposition: ±2 semitones (5 variations)
- Time stretching: 0.95-1.05 (3 variations)
- Velocity scaling: 0.9-1.1 (2 variations)

Effective size: 200h → 6,000h

Regularization:
- Dropout: 0.3 (high)
- Weight decay: 1e-3
- Label smoothing: 0.1
- Stochastic depth: 0.2 (decoder)
- Early stopping: patience=5
```

[Include full hyperparameters from integration doc Section 4.3]

### 3.5 Student Model & Distillation

**Conservative Performance Expectation**:

Previous work (DistilBERT): 97% retention in NLP

**Our expectation**: 75-85% retention (conservative)
- Music is more sensitive to subtle details than text
- Timing (ms-level) and dynamics may not distill well

**Performance Scenarios**:

| Scenario | Teacher | Student | Retention | Probability |
|----------|---------|---------|-----------|-------------|
| Optimistic | 79% | 70% | 88% | 20% |
| Expected | 77% | 65% | 84% | 50% |
| Acceptable | 75% | 60% | 80% | 25% |
| Failure | 75% | <55% | <73% | 5% |

**Contingency**: If <60%, try 2-stage distillation (Teacher→Medium→Student)

---

## 4. Experiments (1.5 pages)

### 4.1 Datasets

[Same as original, but add:]

**Data Augmentation** (to mitigate 200h limitation):
- Pitch: ±2 semitones → 5× augmentation
- Time: 0.95-1.05× → 3× augmentation
- Velocity: ±10% → 2× augmentation
- Effective: 200h → 6,000h equivalent

### 4.2 Baselines (FAIR COMPARISON)

**Key Change**: All baselines fine-tuned on PiJAMA 200h

| Model | Pretrain | Finetune | Architecture |
|-------|----------|----------|--------------|
| Music Transformer | MAESTRO | **+ PiJAMA 200h** | 6L enc, 6L dec |
| Music Informer | MAESTRO | **+ PiJAMA 200h** | 6L enc, 6L dec |
| ImprovNet | ATEPP | **PiJAMA 200h** | 12L enc, 12L dec |
| **JazzFlow-RT Teacher** | ATEPP | **PiJAMA 200h** | 6L enc, 12L dec |
| **JazzFlow-RT Student** | Distilled | N/A | 6L enc, 6L dec |

**Rationale**: Fair comparison requires same training data. Otherwise, gains could be from more jazz data, not architecture.

### 4.3 Evaluation Metrics

**Objective**:
1. Perplexity (lower better)
2. Pitch Class Entropy (target: ~3.42, match real jazz)
3. Groove Similarity (target: >0.85)
4. **Inference Latency**: Model forward pass only
5. **End-to-End Latency**: Full pipeline (MIDI in → MIDI out)

**Subjective** (STRENGTHENED):
```python
# Previous: 20 listeners, 50 samples (too small)
# Revised: 50 listeners, 100 samples per model

Participants:
- 25 jazz experts (music students, professional musicians)
- 25 general listeners (music experience 5+ years)

Samples:
- 100 per model (not 50)
- 30 seconds each
- Same audio quality (same MIDI synthesizer)
- Triple-blind

Procedure:
1. Jazz Recognition:
   - "Is this jazz?" (Yes/No)
   - 100 generated + 100 real jazz (shuffled)

2. Mean Opinion Score (MOS):
   - "Rate musicality (1-5)"
   - "Rate jazz authenticity (1-5)"

3. Preference Test:
   - A/B comparison between models
   - "Which sounds more like jazz?"

Statistical Analysis:
- Inter-rater reliability: Krippendorff's α > 0.7 (required)
- Significance: Paired t-test with Bonferroni correction
- Effect size: Cohen's d
- Report: p-values + 95% confidence intervals

Example:
"JazzFlow-RT Student vs Magenta RT:
 Recognition: 65% vs 55% (diff: +10%, p=0.003, 95% CI: [+4%, +16%], d=0.68)
 → Significantly better (p<0.01, medium effect size)"
```

### 4.4 Reproducibility (Full Details)

[Include all hyperparameters from integration doc Section 4.3]

**Additional**:
```python
# Software versions
PyTorch: 2.0.1
CUDA: 11.8
Python: 3.10
transformers: 4.30.0

# Hardware
4× NVIDIA RTX 3090 (24GB)

# Random seeds
Training: 42
Validation: 43
Test: 44

# Code & Models
GitHub: [https://github.com/[user]/jazzflow-rt]
Models: [HuggingFace Hub links]
Data: [PiJAMA dataset link]
```

---

## 5. Results (2-2.5 pages)

### 5.1 Main Results (ACTUAL, not hypothesized)

**Table 1: Primary Performance Metrics**

| Model | Jazz Recognition | Perplexity | Latency (Inference) | Latency (End-to-End) |
|-------|-----------------|------------|-------------------|---------------------|
| Music Transformer (finetuned) | [X%] | [X] | 5000ms | N/A |
| Music Informer (finetuned) | [X%] | [X] | 4000ms | N/A |
| ImprovNet | 79% | 2.0 | 8000ms | N/A |
| Magenta RealTime | 55% | 3.5 | 50ms | [X]ms |
| **JazzFlow-RT Teacher** | **[X%]** | **[X]** | **[X]ms** | N/A |
| **JazzFlow-RT Student** | **[X%]** | **[X]** | **[X]ms** | **[X]ms** |

**Note**: Results to be filled after experiments (Month 24, Week 2)

**Statistical Validation**:
```
Teacher vs ImprovNet:
- Jazz Recognition: [X]% vs 79% (diff: [±Y]%, p=[Z], 95% CI: [low, high])
- IF p > 0.05: "No significant difference" (claim: "comparable")
- IF p < 0.05: Report actual difference with effect size

Student vs Magenta RT:
- Jazz Recognition: [X]% vs 55% (diff: [±Y]%, p=[Z], 95% CI: [low, high])
- Require p < 0.01 to claim "significantly better"
```

### 5.2 Ablation Studies (COMPREHENSIVE)

**A. ProbSparse Compatibility Test** (CRITICAL)

| Variant | Jazz Recognition | Notes |
|---------|-----------------|-------|
| Standard Attn + Corruption | [X%] | Baseline |
| **ProbSparse + Corruption** | **[X%]** | Proposed |
| ProbSparse + Next-token | [X%] | Music Informer style |

**Analysis**:
```
Degradation = Standard - ProbSparse

IF Degradation < 3%:
    → "ProbSparse and Corruption are compatible.
       Degradation is [X]% (not significant, p=[Y])."
    → Claim success ✅

ELIF Degradation 3-5%:
    → "ProbSparse causes minor degradation ([X]%).
       Trade-off: 21.73% speedup vs [X]% quality loss."
    → Claim partial success ⚠️

ELIF Degradation > 5%:
    → "ProbSparse and Corruption conflict significantly ([X]% loss).
       Hypothesis rejected. Used Standard Attention instead."
    → Report negative result honestly 📊
```

**B. Component Ablations**

| Variant | Jazz Rec. | Perplexity | Latency | Conclusion |
|---------|-----------|------------|---------|-----------|
| Full Model | [X%] | [X] | [X]ms | Baseline |
| w/o ProbSparse | [X%] | [X] | [X]ms | Speedup: [±Y%] |
| w/o Relative Attn | [X%] | [X] | [X]ms | Impact: [±Y%] |
| w/o LSTM | [X%] | [X] | [X]ms | Impact: [±Y%] |
| w/o Corruption | [X%] | [X] | [X]ms | Impact: [±Y%] |
| Decoder 6L (not 12L) | [X%] | [X] | [X]ms | Depth trade-off |

**C. Corruption Function Importance** (Leave-one-out)

| Removed Corruption | Jazz Rec. | Δ from full | Keep? |
|-------------------|-----------|-------------|-------|
| None (full) | [X%] | 0% | Baseline |
| pitch_velocity_mask | [X%] | [±Y%] | ? |
| onset_duration_mask | [X%] | [±Y%] | ? |
| whole_mask | [X%] | [±Y%] | ? |
| permute_pitch | [X%] | [±Y%] | ? |
| fragmentation | [X%] | [±Y%] | ? |
| skyline | [X%] | [±Y%] | ? |
| incorrect_transposition | [X%] | [±Y%] | ? |
| note_modification | [X%] | [±Y%] | ? |
| permute_pitch_velocity | [X%] | [±Y%] | ? |

**Conclusion**: "We find that [X, Y, Z] corruptions contribute most ([A, B, C]% each). [P, Q] have minimal impact (<1%) and can be omitted for efficiency."

**D. Distillation Analysis**

| Variant | Student Jazz Rec. | Retention | Notes |
|---------|------------------|-----------|-------|
| Teacher | [X%] | 100% | Baseline |
| **1-stage distillation** | **[X%]** | **[Y%]** | Proposed |
| 2-stage distillation | [X%] | [Y%] | If needed |
| Trained from scratch | [X%] | [Y%] | No distillation |
| Temperature T=1.0 | [X%] | [Y%] | Lower temp |
| Temperature T=3.0 | [X%] | [Y%] | Higher temp |

**Analysis**:
```
Retention = (Student / Teacher) × 100%

IF Retention ≥ 80%:
    → "Distillation successfully transfers jazz knowledge ([X]% retention)."
    → Compare to NLP (DistilBERT: 97%)
    → "Music is more challenging than NLP ([97-X]% gap)."

IF Retention 70-80%:
    → "Moderate retention ([X]%). Some jazz 'feel' lost in distillation."
    → Discuss what was lost (groove? harmony?)

IF Retention < 70%:
    → "Poor retention ([X]%). Jazz quality does not distill well."
    → Try 2-stage or report negative result
```

**E. Encoder/Decoder Depth Trade-off**

| Config | Enc Layers | Dec Layers | Params | Jazz Rec. | Perplexity |
|--------|-----------|-----------|--------|-----------|------------|
| Baseline | 6 | 6 | 55M | [X%] | [X] |
| **Proposed** | **6** | **12** | **115M** | **[X%]** | **[X]** |
| Deep Enc | 12 | 6 | 115M | [X%] | [X] |
| Maximum | 12 | 12 | 180M | [X%] | [X] |
| Light | 4 | 8 | 70M | [X%] | [X] |

**Conclusion**: "Our 6-12 configuration achieves best quality-efficiency trade-off. Deeper encoder (12-6) does not improve jazz quality, suggesting decoding is the bottleneck."

### 5.3 Latency Analysis (Detailed Breakdown)

**End-to-End Latency Components** (Student):

| Component | Mean (ms) | p95 (ms) | p99 (ms) |
|-----------|-----------|----------|----------|
| MIDI capture | [X] | [X] | [X] |
| Tokenization | [X] | [X] | [X] |
| Chunk processing | [X] | [X] | [X] |
| **Model inference** | **[X]** | **[X]** | **[X]** |
| Sampling | [X] | [X] | [X] |
| Detokenization | [X] | [X] | [X] |
| MIDI output | [X] | [X] | [X] |
| **Total** | **[X]** | **[X]** | **[X]** |

**Real-Time Capability**:
```
IF p95 < 100ms:
    → "Achieves real-time performance ([X]ms p95 latency)."
    → Claim success ✅

IF 100ms < p95 < 150ms:
    → "Near-real-time performance ([X]ms p95).
       Perceptually acceptable for most interactive applications."
    → Claim partial success ⚠️

IF p95 > 150ms:
    → "Does not achieve real-time ([X]ms p95).
       Suitable for offline generation only."
    → Report limitation honestly 📊
```

### 5.4 Subjective Evaluation (Statistical Rigor)

**Jazz Style Recognition Test**:

| Model | Recognition Rate | 95% CI | vs ImprovNet (p-value) | Effect Size (d) |
|-------|-----------------|--------|----------------------|----------------|
| Real Jazz | 91% | [X, X] | N/A | N/A |
| ImprovNet | 79% | [X, X] | Baseline | Baseline |
| **Teacher** | **[X%]** | **[X, X]** | **p=[X]** | **d=[X]** |
| **Student** | **[X%]** | **[X, X]** | **p=[X]** | **d=[X]** |
| Music Transformer | [X%] | [X, X] | p=[X] | d=[X] |
| Magenta RT | 55% | [X, X] | p=[X] | d=[X] |

**Inter-Rater Reliability**: Krippendorff's α = [X] (>0.7 required)

**Mean Opinion Score** (5-point Likert):

| Model | Musicality | Jazz Authenticity | Overall |
|-------|------------|------------------|---------|
| ImprovNet | 4.6 | 4.5 | 4.6 |
| **Teacher** | **[X]** | **[X]** | **[X]** |
| **Student** | **[X]** | **[X]** | **[X]** |
| Magenta RT | 3.0 | 2.8 | 3.0 |

**Preference Test** (% prefer JazzFlow-RT):

| Comparison | Preference | p-value |
|-----------|-----------|---------|
| Teacher vs ImprovNet | [X%] | p=[X] |
| Student vs Magenta RT | [X%] | p=[X] |
| Student vs Teacher | [X%] | p=[X] |

---

## 6. Discussion (1-1.5 pages)

### 6.1 Key Findings

**Finding 1: ProbSparse + Corruption Compatibility**
```
IF Compatible:
    "Our experiments demonstrate that ProbSparse Attention and
     corruption-refinement are compatible, with only [X]% quality loss.
     This suggests efficient attention mechanisms CAN work for music
     generation, contrary to common assumption that music requires
     full O(L²) attention."

IF Incompatible:
    "Our experiments reveal a fundamental incompatibility between
     ProbSparse Attention and corruption-refinement ([X]% degradation).
     ProbSparse skips corrupted positions, preventing effective
     restoration. This negative result is valuable: efficient attention
     designed for smooth time-series does NOT transfer to discrete,
     rhythmic music."
```

**Finding 2: Distillation Retention Rate**
```
"We find that jazz quality distills at [X]% retention, compared to
 97% for NLP (DistilBERT). This gap suggests musical 'feel' (timing,
 dynamics) is harder to transfer than semantic knowledge. Specifically,
 [component X] degrades most ([Y]%), suggesting future work should
 focus on [...]."
```

**Finding 3: Real-Time Trade-Offs**
```
IF Successful:
    "Our student model demonstrates that >65% jazz recognition at
     <100ms latency is achievable, resolving the quality-latency trade-off."

IF Partial:
    "While we achieve [X]ms latency, jazz quality ([Y]%) is lower than
     hoped. We identify a fundamental trade-off: [analysis]. Applications
     requiring >70% quality should use teacher (offline), while interactive
     applications can use student despite lower quality."
```

### 6.2 Comparison to Hypotheses

| Hypothesis | Result | Validated? |
|-----------|--------|-----------|
| H1: ProbSparse preserves quality | [X% degradation] | [Yes/No/Partial] |
| H2: ProbSparse + Corruption compatible | [Compatible/Incompatible] | [Yes/No] |
| H3: Distillation 75-85% retention | [X% retention] | [Yes/No/Partial] |
| H4: <100ms end-to-end latency | [X ms p95] | [Yes/No/Partial] |

**Honest Assessment**:
```
"Of our four hypotheses, [X] were validated, [Y] partially validated,
 and [Z] rejected. This demonstrates [...]."
```

### 6.3 Limitations

1. **Dataset Size**: 200h PiJAMA is small (mitigated by augmentation)
2. **Genre-Specific**: Trained only on jazz piano solos
3. **Evaluation**: Subjective metrics depend on listener expertise
4. **Computational Cost**: Teacher training requires 4× RTX 3090 GPUs

### 6.4 When Our Approach Works/Fails

**Works Best For**:
- [Based on results, e.g., "Harmonization tasks (94% success)"]
- [e.g., "Medium-tempo swing (as opposed to fast bebop)"]

**Struggles With**:
- [Based on results, e.g., "Cross-genre improvisation (only 82% success)"]
- [e.g., "Very fast passages (>300 BPM)"]

### 6.5 Future Work

1. **Multi-instrument**: Extend to full jazz ensemble
2. **Larger dataset**: Train on combined PiJAMA + Doug McKenzie (300h)
3. **Controllable generation**: Add swing amount, harmonic density controls
4. **Reinforcement learning**: Optimize student for user preference
5. **Alternative efficient attention**: Try other mechanisms (e.g., Linear Attention, Performer)

---

## 7. Conclusion (0.5 page)

**Conservative Version**:

We investigated whether ProbSparse Attention, corruption-refinement learning, and knowledge distillation can be combined for real-time jazz generation. Our teacher model [achieved/struggled to achieve] [X%] jazz recognition while [maintaining/sacrificing] the 21.73% speedup from ProbSparse Attention. Our student model [achieved/struggled to achieve] [X%] recognition at [Y]ms end-to-end latency.

**Key contributions**:
1. **Empirical validation** of ProbSparse Attention for music [success/failure]
2. **Comprehensive ablation studies** revealing [key insights]
3. **Fair baseline comparisons** with statistical rigor
4. **Open-source release** enabling reproducibility

**If successful** (>65% student, <100ms latency):
"Our results demonstrate that high-quality real-time jazz generation is achievable, opening new possibilities for interactive music systems."

**If partial success** (60-65% student, 100-150ms):
"Our results reveal fundamental challenges in distilling musical quality while maintaining low latency. We provide detailed analysis of trade-offs to guide future work."

**If failure** (<60% student or >150ms):
"While our system did not achieve real-time high-quality generation, our negative results provide valuable insights: [1] ProbSparse does not work well for jazz, [2] musical 'feel' does not distill effectively, [3] system overhead dominates latency. These findings will inform future research directions."

**Honest Science**:
"We report our results transparently—both successes and failures—to advance the field's understanding of efficient music generation."

---

## 8. Reproducibility Checklist

✅ Code: GitHub repository with full implementation
✅ Models: Pretrained checkpoints on HuggingFace Hub
✅ Data: PiJAMA dataset access instructions
✅ Hyperparameters: Complete specification (Section 4.4)
✅ Seeds: Random seeds for training/validation/test
✅ Hardware: Detailed GPU/CPU specifications
✅ Software: Library versions (PyTorch, CUDA, etc.)
✅ Evaluation: Listening test protocol and materials
✅ Statistics: Raw data and analysis scripts
✅ Negative Results: Documented failed experiments

---

## 9. Supplementary Materials

**Website**: [https://jazzflow-rt.github.io]
- Audio examples (generated vs real jazz)
- Interactive demo
- Ablation study visualizations
- Failed experiment documentation

**Code**: [https://github.com/[user]/jazzflow-rt]
- Complete implementation
- Training scripts
- Evaluation scripts
- Pretrained models

**Data**:
- Listening test raw responses
- Statistical analysis notebooks
- Generated samples for each model

---

## 10. Realistic Outcome Scenarios

### Scenario A: Strong Success (20% probability)

**Results**:
- Teacher: 78% jazz recognition (within 1% of ImprovNet)
- Student: 68% recognition, 85ms p95 latency
- ProbSparse compatible (<3% degradation)
- 84% distillation retention

**Paper Claim**:
"First real-time jazz generation system achieving 68% style recognition at 85ms latency through novel hybrid architecture. Empirically validates ProbSparse Attention for music with minimal quality loss."

**Likely Outcome**: **Accept at ISMIR 2026** ✅

---

### Scenario B: Moderate Success (50% probability)

**Results**:
- Teacher: 75% jazz recognition (-4% from ImprovNet, ProbSparse causes slight degradation)
- Student: 63% recognition, 115ms p95 latency
- ProbSparse causes 4% degradation (marginal compatibility)
- 78% distillation retention

**Paper Claim**:
"Investigates combining ProbSparse Attention with corruption-refinement for jazz generation. Achieves 75% quality offline and 63% at 115ms latency. Reveals trade-offs: ProbSparse provides speedup but with quality cost. Distillation retention (78%) lower than NLP (97%), suggesting musical 'feel' is challenging to transfer."

**Likely Outcome**: **Accept at ISMIR 2026** (Solid empirical work) ✅

---

### Scenario C: Weak Success (25% probability)

**Results**:
- Teacher: 72% jazz recognition (ProbSparse incompatible, switched to Standard)
- Student: 58% recognition, 135ms p95 latency
- ProbSparse incompatible (7% degradation, hypothesis rejected)
- 67% distillation retention

**Paper Claim**:
"We investigate whether efficient attention mechanisms designed for time-series forecasting transfer to music generation. Our experiments reveal that ProbSparse Attention significantly degrades jazz quality (7%), suggesting it does not capture syncopation patterns. We provide detailed ablation studies and pivot to Standard Attention, achieving 72% recognition. Distillation to real-time model achieves 58% at 135ms latency—modest improvement over prior work (55%, 50ms) but reveals fundamental challenges in distilling musical quality."

**Likely Outcome**: **Borderline at ISMIR 2026** (Weak accept or reject, but good science) ⚠️

---

### Scenario D: Failure (5% probability)

**Results**:
- Teacher: <70% (severe issues)
- Student: <55% (worse than Magenta RT)
- Multiple hypotheses rejected

**Action**:
- Pivot to workshop paper: "When Does ProbSparse Attention Fail? A Case Study in Jazz Generation"
- Or: ISMIR 2027 with more investigation

---

## Final Meta-Comment

**This paper outline represents RESPONSIBLE SCIENCE**:

✅ Clear hypotheses stated upfront
✅ Success criteria defined before experiments
✅ Multiple outcome scenarios prepared
✅ Commitment to reporting negative results
✅ Fair baseline comparisons
✅ Statistical rigor
✅ Full reproducibility

**Whether we achieve 79% or 58%, whether latency is 50ms or 150ms, we will report honestly and advance the field's understanding.**

---

**End of Revised ISMIR 2026 Paper Outline**

**Status**: Ready for experimentation (Months 20-25)
**Next Steps**: Implement, measure, report truthfully
