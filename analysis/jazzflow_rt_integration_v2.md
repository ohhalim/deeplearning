# JazzFlow-RT: Integration Strategy (REVISED)

**Author**: Research Team (Self-Critical Review Applied)
**Date**: 2025-11-18
**Version**: 2.0 (Conservative & Realistic)
**Status**: Research Proposal for 2026 ISMIR Paper

---

## ⚠️ IMPORTANT: This is a Research Proposal

**This document presents HYPOTHESES, not proven results.**

All performance claims are:
- ✅ **Hypothesized targets** based on component capabilities
- ❌ **NOT guaranteed outcomes** - experimentation required
- ⚠️ **Subject to validation** through rigorous testing

**Success is NOT guaranteed.** Multiple risk mitigation strategies are provided.

---

## Executive Summary

**JazzFlow-RT** proposes combining three state-of-the-art models:

1. **Music Informer** (Nature 2025): ProbSparse Attention for efficiency
2. **ImprovNet** (Nature 2025): Corruption-refinement for jazz quality
3. **Magenta RealTime** (Google 2025): Streaming architecture for real-time

**Research Question**: Can we achieve both high jazz quality AND real-time performance?

**Hypothesized Performance** (to be validated):

| Model | Jazz Recognition | Latency | Status |
|-------|-----------------|---------|--------|
| Teacher (Offline) | **Target: 75-79%** | 3.5s | Optimistic if ProbSparse doesn't degrade quality |
| Student (Real-Time) | **Target: 60-70%** | <100ms end-to-end | Conservative estimate for distillation |

**Novel Contribution** (if successful):
- First system achieving >65% jazz recognition at <100ms latency
- Hybrid architecture combining three complementary techniques
- Empirical validation of ProbSparse + Corruption compatibility

**Target Venue**: ISMIR 2026

---

## 1. Research Objectives & Risks

### 1.1 Primary Research Questions

**RQ1**: Does ProbSparse Attention preserve jazz quality?
- **Hypothesis**: 21.73% speedup with minimal quality loss (<5%)
- **Risk**: ProbSparse may miss syncopation patterns → quality degradation
- **Validation**: Ablation study (Section 6.2)

**RQ2**: Are Corruption-Refinement and ProbSparse compatible?
- **Hypothesis**: Can combine both techniques successfully
- **Risk**: ProbSparse may skip corrupted positions → poor reconstruction
- **Validation**: Dedicated compatibility test (Section 6.3)

**RQ3**: How much jazz knowledge transfers via distillation?
- **Hypothesis**: 75-85% retention (conservative estimate)
- **Risk**: Jazz "feel" may not distill well → only 60-70% retention
- **Validation**: Measured after distillation (Month 23)

**RQ4**: Can we achieve <100ms end-to-end latency?
- **Hypothesis**: Streaming + KV caching + optimizations → Yes
- **Risk**: System overhead (MIDI I/O, OS scheduling) may dominate
- **Validation**: Realistic end-to-end measurement (Section 7.3)

### 1.2 Success Criteria

**Minimum Viable Success**:
- Teacher: ≥75% jazz recognition (within 5% of ImprovNet)
- Student: ≥60% jazz recognition (better than Magenta RT's 55%)
- Student: <100ms p95 end-to-end latency

**Stretch Goals**:
- Teacher: 79% (match ImprovNet)
- Student: 70% (retain 88% of teacher quality)
- Student: <50ms p95 end-to-end latency

**Failure Criteria** (triggers pivot):
- Teacher: <70% (ProbSparse degrading quality too much)
- Student: <55% (distillation not working)
- Student: >150ms p95 latency (not real-time capable)

---

## 2. Architecture Design

### 2.1 Teacher Model: High-Quality Offline

**Design Rationale**:
- **Encoder**: Music Informer style (ProbSparse for efficiency)
- **Decoder**: ImprovNet style (deep for expressiveness)
- **Training**: Corruption-refinement (for jazz quality)

**Architecture**:
```python
class JazzFlowRTTeacher(nn.Module):
    def __init__(self):
        # Encoder: Music Informer style
        self.encoder = MusicInformerEncoder(
            num_layers=6,
            d_model=768,        # Match ImprovNet
            num_heads=12,
            d_ff=3072,
            d_lstm=1024,
            use_prob_sparse=True,   # ⚠️ To be validated
            use_relative=True,
            sampling_factor=5
        )

        # Decoder: ImprovNet style
        self.decoder = ImprovNetDecoder(
            num_layers=12,      # Deep for expressiveness
            d_model=768,
            num_heads=12,
            d_ff=3072
        )
```

**Hypothesized Performance**:
- Jazz Recognition: 75-79% (if ProbSparse doesn't degrade)
- Perplexity: 2.0-2.2
- Latency: ~3.5 seconds (512 tokens)
- Encoding speedup: 21.73% (from ProbSparse)

**Risk Mitigation**:
- If ProbSparse degrades quality >5%: Use Standard Attention instead
- If 12-layer decoder overfits: Reduce to 8-10 layers
- If training unstable: Add stronger regularization

### 2.2 Student Model: Real-Time

**Design Rationale**:
- Lightweight for speed (40M params vs teacher's 115M)
- Standard attention (simpler than ProbSparse for distillation)
- KV caching for streaming

**Architecture**:
```python
class JazzFlowRTStudent(nn.Module):
    def __init__(self):
        self.encoder = StreamingEncoder(
            num_layers=6,
            d_model=512,        # Smaller than teacher
            num_heads=8,
            d_ff=2048,
            chunk_size=10       # 10ms chunks
        )

        self.decoder = StreamingDecoder(
            num_layers=6,       # Shallower than teacher
            d_model=512,
            num_heads=8,
            d_ff=2048,
            use_kv_cache=True
        )
```

**Hypothesized Performance** (Conservative):
- Jazz Recognition: 60-70% (75-85% of teacher's 75-79%)
- Perplexity: 2.5-3.0
- Inference latency: 20-30ms per chunk
- **End-to-end latency: 80-120ms** (including all overhead)

**Risk Mitigation**:
- If <60% recognition: Try 2-stage distillation (Medium → Student)
- If >150ms latency: Optimize or reduce chunk size
- If quality too low: Use teacher for offline, student for preview only

---

## 3. Critical Compatibility Test: ProbSparse + Corruption

### 3.1 Potential Conflict

**Concern**: ProbSparse may conflict with corruption-refinement.

**ProbSparse Assumption**: Most queries are uninformative → skip them
**Corruption Assumption**: All positions need restoration → all are informative

**Example Conflict**:
```python
# Clean sequence
Clean: [100, 560, 633, 900, 150, 564, 638, 910, 200, 567, 640, 920]

# After corruption (50% masked)
Corrupted: [100, MASK, MASK, 900, 150, MASK, MASK, 910, 200, MASK, MASK, 920]

# ProbSparse computes M (importance) for each query
# If MASK positions have low M → they get skipped
# → Cannot restore masked positions!
```

### 3.2 Hypothesis & Validation Plan

**Hypothesis H1** (Optimistic): MASK tokens have high M-score → ProbSparse doesn't skip them
**Hypothesis H2** (Pessimistic): MASK tokens have low M-score → ProbSparse skips → quality degrades

**Validation Experiment** (Month 20, Week 1-2):
```python
# Test 1: Standard Attention + Corruption
model_baseline = Encoder(use_prob_sparse=False) + Decoder()
train(model_baseline, corruption_data)
eval_baseline = evaluate(model_baseline)  # Expect: ~79%

# Test 2: ProbSparse Attention + Corruption
model_probsparse = Encoder(use_prob_sparse=True) + Decoder()
train(model_probsparse, corruption_data)
eval_probsparse = evaluate(model_probsparse)  # Expect: 74-79%

# Decision:
if eval_probsparse < eval_baseline - 5%:
    print("⚠️ ProbSparse degrades quality too much")
    print("→ PIVOT: Use Standard Attention in encoder")
else:
    print("✅ ProbSparse + Corruption compatible")
    print("→ PROCEED with hybrid architecture")
```

**Contingency Plan**:
- If incompatible: Use Standard Attention (sacrifice speed for quality)
- Alternative: Hybrid encoder (ProbSparse in early layers, Standard in late layers)

---

## 4. Training Strategy

### 4.1 Dataset & Augmentation

**Primary Dataset**: PiJAMA (200 hours jazz)

**Problem**: 200h is small for 115M parameters → overfitting risk

**Solution: Aggressive Data Augmentation**
```python
class JazzDataAugmentation:
    def augment(self, midi_sequence):
        augmented = []

        # 1. Pitch transposition (12 variations)
        for semitones in range(-6, 6):  # ±5 semitones
            transposed = transpose(midi_sequence, semitones)
            augmented.append(transposed)

        # 2. Time stretching (5 variations)
        for ratio in [0.95, 0.975, 1.0, 1.025, 1.05]:
            stretched = time_stretch(midi_sequence, ratio)
            augmented.append(stretched)

        # 3. Velocity scaling (3 variations)
        for scale in [0.9, 1.0, 1.1]:
            scaled = velocity_scale(midi_sequence, scale)
            augmented.append(scaled)

        return augmented

# Effective dataset size:
# 200h × 12 (pitch) × 5 (time) × 3 (velocity) = 36,000h equivalent
# (In practice, use subset to avoid explosion)
```

**Practical Augmentation**:
- Pitch: ±2 semitones (5 variations) → 1,000h
- Time: 0.95-1.05 (3 variations) → 3,000h
- Velocity: 0.9-1.1 (2 variations) → 6,000h

**Regularization**:
- Dropout: 0.3 (high for small dataset)
- Weight decay: 1e-3
- Label smoothing: 0.1
- Early stopping: patience=5 epochs
- Stochastic depth: 0.2 (decoder only)

### 4.2 Three-Phase Training

**Phase 1: Pretrain Encoder (Month 20)**
- Dataset: ATEPP (1000h classical)
- Task: Next-token prediction (simple pretraining)
- Epochs: 15-20 (until validation plateaus)
- **Goal**: Learn general music structure

**Phase 2: Pretrain Decoder (Month 21)**
- Dataset: ATEPP (1000h classical)
- Task: Corruption-refinement
- Freeze encoder, train decoder only
- Epochs: 15-20
- **Goal**: Learn refinement capability

**Phase 3: Fine-tune on Jazz (Month 22)**
- Dataset: PiJAMA (200h → 6,000h with augmentation)
- Task: All 5 ImprovNet tasks (CGI, IGI, continuation, infilling, harmonization)
- Unfreeze encoder, joint training
- Epochs: 30-40 (with early stopping)
- **Goal**: Specialize in jazz

**Expected Timeline**:
- Phase 1: 3 weeks (encoder pretrain + ProbSparse validation)
- Phase 2: 3 weeks (decoder pretrain + corruption validation)
- Phase 3: 4 weeks (jazz finetuning + ablations)
- Buffer: 2 weeks (unexpected issues)

### 4.3 Training Hyperparameters (Full Reproducibility)

```python
# Optimizer
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-4,           # Initial learning rate
    betas=(0.9, 0.98), # β₁, β₂
    eps=1e-8,
    weight_decay=1e-3  # Regularization
)

# Learning rate schedule
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,
    pct_start=0.1,      # 10% warmup
    anneal_strategy='cos'
)

# Regularization
dropout = 0.3           # High due to small dataset
label_smoothing = 0.1
gradient_clip_norm = 1.0
stochastic_depth_rate = 0.2  # Decoder only

# Batch configuration
per_gpu_batch_size = 4  # Due to long sequences
num_gpus = 4
gradient_accumulation = 2
effective_batch_size = 4 × 4 × 2 = 32

# Sequence configuration
max_seq_len = 512       # Aria tokens
padding = 'right'       # Right-pad to 512

# Random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hardware
device = 'cuda'
mixed_precision = True  # FP16 via torch.cuda.amp
cudnn_benchmark = True
```

---

## 5. Knowledge Distillation

### 5.1 Conservative Performance Expectation

**Previous Claim** (too optimistic): 70% (88% retention)
**Revised Claim** (realistic): 60-70% (75-88% retention)

**Rationale**:
- NLP distillation (DistilBERT): 97% retention - but NLP is discrete
- Music distillation: More challenging due to continuous timing/dynamics
- Jazz "feel": Subtle and may not distill well

**Performance Scenarios**:

| Scenario | Teacher | Student | Retention | Probability |
|----------|---------|---------|-----------|-------------|
| **Best case** | 79% | 70% | 88% | 20% |
| **Expected** | 77% | 65% | 84% | 50% |
| **Acceptable** | 75% | 60% | 80% | 25% |
| **Failure** | 75% | <55% | <73% | 5% (pivot needed) |

### 5.2 Distillation Strategy

**Standard Distillation**:
```python
# Temperature
tau = 2.0  # Soften teacher distribution

# Loss
soft_loss = KL_divergence(
    student_logits / tau,
    teacher_logits / tau
) * (tau ** 2)

hard_loss = CrossEntropy(student_logits, ground_truth)

total_loss = 0.7 * soft_loss + 0.3 * hard_loss
```

**If Standard Distillation Fails (<60%): 2-Stage Distillation**
```python
# Stage 1: Teacher → Medium model
Medium = Model(params=70M)  # Between Teacher (115M) and Student (40M)
distill(Teacher → Medium)  # Expect: 72-75%

# Stage 2: Medium → Student
distill(Medium → Student)  # Expect: 65-70%

# Total retention: 90% × 90% = 81% (better than 1-stage)
```

### 5.3 Task-Specific Distillation (Fallback)

If general distillation fails, distill per task:

| Task | Difficulty | Distillation Success Probability |
|------|-----------|--------------------------------|
| Harmonization | Easy | 90% (vertical structure) |
| IGI | Medium | 80% (same genre) |
| Continuation | Medium | 75% (local context) |
| Infilling | Hard | 65% (bidirectional) |
| CGI | Very Hard | 50% (cross-genre) |

**Strategy**: Focus on easy tasks, use teacher for hard tasks in deployment.

---

## 6. Evaluation & Validation

### 6.1 Baseline Comparison (FAIR)

**Problem**: Unfair to compare our jazz-finetuned model to non-finetuned baselines

**Solution**: Finetune ALL baselines on same PiJAMA 200h

| Model | Architecture | Pretrain | Finetune | Expected Jazz Rec. |
|-------|-------------|----------|----------|-------------------|
| Music Transformer | 6L enc, 6L dec | MAESTRO | None | 64% |
| Music Transformer | 6L enc, 6L dec | MAESTRO | **PiJAMA 200h** | **~72%** |
| Music Informer | 6L enc, 6L dec | MAESTRO | None | 68% |
| Music Informer | 6L enc, 6L dec | MAESTRO | **PiJAMA 200h** | **~75%** |
| ImprovNet | 12L enc, 12L dec | ATEPP | PiJAMA 200h | 79% |
| **JazzFlow-RT Teacher** | 6L enc, 12L dec | ATEPP | PiJAMA 200h | **Target: 75-79%** |
| **JazzFlow-RT Student** | 6L enc, 6L dec | Distilled | N/A | **Target: 60-70%** |

**Claim** (if successful):
"Even with fair comparison (all models finetuned on PiJAMA), our Teacher approaches ImprovNet quality while being 21.73% faster in encoding. Our Student achieves real-time performance while maintaining higher quality than non-finetuned baselines."

### 6.2 Ablation Studies (Comprehensive)

**A. Component Ablations**

| Variant | Jazz Rec. (Hypothesis) | Latency | Conclusion |
|---------|----------------------|---------|-----------|
| Full Model | 75-79% | 3.5s | Baseline |
| w/o ProbSparse | 75-79% | 4.5s | +29% latency, same quality → ProbSparse worth it IF compatible |
| w/o Relative Attn | 70-75% | 3.5s | -5% quality → Relative helps |
| w/o LSTM | 67-73% | 3.2s | -8% quality → LSTM important |
| w/o Corruption | 56-65% | 3.5s | -19% quality → Corruption critical |
| Decoder 6L (not 12L) | 68-75% | 2.5s | -7% quality → Deep decoder helps |

**B. ProbSparse + Corruption Compatibility** (CRITICAL)

| Variant | Jazz Rec. | Notes |
|---------|-----------|-------|
| Standard + Corruption | 79% (ImprovNet baseline) | Reference |
| ProbSparse + Corruption | **74-79%** | Test compatibility |
| ProbSparse + Next-token | 70-75% | Music Informer style |

**Decision Tree**:
```
IF ProbSparse + Corruption ≥ 75%:
    → Use hybrid (ProbSparse encoder + Corruption training)
ELIF ProbSparse + Corruption < 70%:
    → Conflict confirmed, use Standard Attention
ELSE (70-75%):
    → Marginal, try hybrid approach (some layers ProbSparse, some Standard)
```

**C. Corruption Function Importance** (Leave-one-out)

Test which corruptions are most important:

| Removed Corruption | Expected Impact | Decision |
|-------------------|----------------|----------|
| pitch_velocity_mask | -5% (harmonization degrades) | Keep |
| onset_duration_mask | -4% (rhythm degrades) | Keep |
| whole_mask | -3% (infilling degrades) | Keep |
| permute_pitch | -2% | Keep |
| fragmentation | -6% (continuation degrades) | Keep |
| skyline | -1% (only affects harmonization task) | Optional |
| incorrect_transposition | -2% | Keep |
| note_modification | -3% | Keep |
| permute_pitch_velocity | -1% | Optional |

**Result**: Use top 7 corruptions, drop 2 least important (saves compute).

**D. Distillation Temperature Sweep**

| Temperature | Student Jazz Rec. (Hypothesis) |
|-------------|-------------------------------|
| T=1.0 | 62-68% |
| T=1.5 | 63-69% |
| **T=2.0** | **64-70%** (best) |
| T=3.0 | 61-66% |
| T=4.0 | 58-64% |

**E. Encoder/Decoder Depth Trade-off**

| Encoder Layers | Decoder Layers | Params | Jazz Rec. (Hyp.) | Notes |
|---------------|---------------|--------|-----------------|-------|
| 6 | 6 | 55M | 68-72% | Baseline (Music Informer) |
| **6** | **12** | **115M** | **75-79%** | Proposed (balanced) |
| 12 | 6 | 115M | 70-75% | Alternative (deep encoding) |
| 12 | 12 | 180M | 77-80% | Maximum (overfit risk) |
| 4 | 10 | 85M | 72-77% | Lighter (less overfit) |

**Decision**: If 6-12 overfits, try 4-10 or 6-8.

### 6.3 Latency Measurement (Realistic)

**Two Metrics Required**:

1. **Inference Latency** (model only)
   - Target: <25ms (p50), <40ms (p99)
   - Measure: Forward pass only, 1000 iterations

2. **End-to-End Latency** (full pipeline)
   - Target: <100ms (p50), <150ms (p99)
   - Measure: MIDI input → MIDI output, realistic conditions

**End-to-End Pipeline**:
```python
def measure_end_to_end_latency():
    latencies = []

    for i in range(1000):
        t_start = time.perf_counter()

        # 1. MIDI input capture (simulate 10ms buffer)
        midi_events = capture_midi(duration_ms=10)  # ~10ms

        # 2. Tokenization
        tokens = tokenizer.encode(midi_events)  # ~5ms

        # 3. Chunk processing
        chunk = chunk_processor.process(tokens)  # ~2ms

        # 4. Model inference
        logits, cache = model.forward(chunk, cache)  # ~25ms (target)

        # 5. Sampling
        output_tokens = sample(logits, top_k=40)  # ~3ms

        # 6. Detokenization
        output_midi = tokenizer.decode(output_tokens)  # ~5ms

        # 7. MIDI output
        send_midi(output_midi)  # ~10ms

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

    return {
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'mean': np.mean(latencies)
    }

# Expected:
# p50: 60-80ms
# p95: 90-120ms
# p99: 100-150ms
```

### 6.4 Statistical Validation

**Subjective Evaluation Requirements**:

**Listening Test Design**:
```python
# Participants
num_listeners = 50  # Not 20 (too small)
  - 25 jazz experts (music students, jazz musicians)
  - 25 general listeners (music experience 5+ years)

# Samples
samples_per_model = 100  # Not 50 (too small)
total_samples = 100 (generated) + 100 (real jazz) = 200

# Procedure
1. Sample preparation:
   - Same length (30 seconds)
   - Same audio quality (same MIDI synthesizer)
   - Triple-blind (generator, evaluator, analyst all blinded)

2. Randomization:
   - Shuffle order
   - Each listener rates subset (40 samples to avoid fatigue)
   - Balanced across models

3. Questions:
   - "Is this jazz?" (Yes/No) → Jazz Recognition Rate
   - "Rate musicality (1-5)" → Mean Opinion Score
   - "Rate jazz authenticity (1-5)" → Jazz Quality Score

# Statistical Analysis
1. Inter-rater reliability:
   - Krippendorff's α > 0.7 (required)

2. Significance testing:
   - Paired t-test (each model vs baseline)
   - Bonferroni correction for multiple comparisons
   - Report p-values and 95% confidence intervals

3. Effect size:
   - Cohen's d for meaningful difference
   - d > 0.5 considered meaningful
```

**Example Report**:
```
JazzFlow-RT Teacher vs ImprovNet:
  - Jazz Recognition: 77% vs 79% (diff: -2%, p=0.34, CI: [-6%, +2%])
  - Not significantly different (p > 0.05)
  - Cohen's d = 0.12 (small effect)
  → Claim: "Comparable to ImprovNet" ✅

JazzFlow-RT Student vs Magenta RT:
  - Jazz Recognition: 65% vs 55% (diff: +10%, p=0.003, CI: [+4%, +16%])
  - Significantly better (p < 0.01)
  - Cohen's d = 0.68 (medium effect)
  → Claim: "Significantly better than Magenta RT" ✅
```

---

## 7. Risk Mitigation & Contingency Plans

### 7.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **ProbSparse degrades quality >5%** | 40% | High | Use Standard Attention |
| **ProbSparse + Corruption incompatible** | 30% | High | Hybrid encoder or pivot |
| **200h dataset → severe overfit** | 60% | Medium | Aggressive augmentation + regularization |
| **Distillation <60% retention** | 35% | Medium | 2-stage distillation |
| **End-to-end latency >150ms** | 25% | Medium | Optimize or relax real-time claim |
| **Teacher <75% jazz recognition** | 20% | High | Reduce model size or adjust claim |

### 7.2 Contingency Plans

**Contingency 1: ProbSparse Fails**
```
IF Teacher with ProbSparse < 75% jazz recognition:
    → Switch to Standard Attention in encoder
    → Claim: "Combines ImprovNet training with deep decoder"
    → Still novel contribution (different from ImprovNet)
    → Speed claim: Remove (no longer faster)
```

**Contingency 2: Distillation Fails (<60%)**
```
IF Student < 60% jazz recognition:
    Option A: 2-stage distillation
    Option B: Larger student (60M params instead of 40M)
    Option C: Task-specific deployment
        - Use student only for harmonization (easier)
        - Use teacher for CGI/IGI (harder)
    Option D: Pivot to "preview mode"
        - Student generates rough draft (fast)
        - Teacher refines offline (high quality)
```

**Contingency 3: Not Real-Time (<100ms fails)**
```
IF End-to-end latency >150ms p99:
    Option A: Relax claim to "near-real-time" (<200ms)
    Option B: Smaller chunks (5ms instead of 10ms)
    Option C: More aggressive optimization (INT8 quantization)
    Option D: Offline-only paper (remove real-time claim)
```

**Contingency 4: Dataset Too Small (Severe Overfit)**
```
IF Train-val gap >0.5 perplexity:
    Option A: Stronger regularization (dropout 0.5)
    Option B: Smaller model (encoder 4L, decoder 8L)
    Option C: Add Doug McKenzie dataset (+307 pieces)
    Option D: Stop early (epoch 20 instead of 50)
```

---

## 8. Revised Timeline (Realistic with Buffer)

| Month | Week | Task | Deliverable | Risk Check |
|-------|------|------|-------------|-----------|
| **20** | 1-2 | Encoder pretrain + **ProbSparse validation** | Encoder checkpoint | ProbSparse quality check |
| | 3-4 | **ProbSparse ablation study** | Decision: keep or pivot | GO/NO-GO decision |
| **21** | 1-2 | Decoder pretrain (corruption) | Decoder checkpoint | Corruption effectiveness |
| | 3-4 | **Corruption ablation study** | Optimal corruption set | Finalize training recipe |
| **22** | 1-2 | Jazz finetune (early phase) | Teacher v0.1 | Overfitting monitoring |
| | 3 | Jazz finetune (late phase) | Teacher v1.0 | Final teacher model |
| | 4 | **Teacher evaluation & ablations** | Teacher results | Success criteria check |
| **23** | 1 | Implement streaming architecture | Student architecture | - |
| | 2-3 | Distillation (1-stage) | Student v1.0 | Distillation quality check |
| | 4 | **If needed: 2-stage distillation** | Student v2.0 (if needed) | Fallback if <60% |
| **24** | 1 | End-to-end latency optimization | Optimized student | Real-time validation |
| | 2 | Comprehensive evaluation | All metrics | - |
| | 3 | Baseline comparisons + statistics | Fair comparison results | - |
| | 4 | Paper writing (draft) | ISMIR draft | - |
| **25** | 1-2 | **BUFFER: Handle unexpected issues** | - | Contingency time |
| | 3-4 | Paper revision & submission | Final paper | ISMIR 2026 submission |

**Total Time**: 6 months (Month 20-25), not 5
**Buffer**: 2 weeks built-in for unexpected issues

---

## 9. Expected Outcomes (Realistic Scenarios)

### Scenario A: Best Case (20% probability)

**Results**:
- Teacher: 79% jazz recognition, 2.0 perplexity, 3.5s latency
- Student: 70% jazz recognition, 2.5 perplexity, 80ms p95 latency
- ProbSparse compatible with corruption (no quality loss)
- 1-stage distillation successful (88% retention)

**Paper Claim**:
"First real-time jazz generation system achieving 70% style recognition at 80ms latency while matching SOTA (79%) in offline mode through novel hybrid architecture combining ProbSparse Attention and corruption-refinement learning."

**Impact**: Strong accept at ISMIR 2026

---

### Scenario B: Expected Case (50% probability)

**Results**:
- Teacher: 77% jazz recognition, 2.1 perplexity, 3.5s latency
  - ProbSparse causes slight quality loss (-2%)
- Student: 65% jazz recognition, 2.7 perplexity, 110ms p95 latency
  - 1-stage distillation moderate (84% retention)
  - End-to-end latency slightly higher than target

**Paper Claim**:
"JazzFlow-RT achieves 65% jazz style recognition at 110ms latency through knowledge distillation, significantly outperforming prior real-time models (55%) while maintaining near-SOTA offline quality (77% vs 79%)."

**Impact**: Accept at ISMIR 2026

---

### Scenario C: Acceptable Case (25% probability)

**Results**:
- Teacher: 75% jazz recognition, 2.2 perplexity, 4.5s latency
  - ProbSparse incompatible → used Standard Attention instead
  - Still better than Music Informer baseline
- Student: 60% jazz recognition, 3.0 perplexity, 130ms p95 latency
  - Required 2-stage distillation
  - Just meets minimum criteria

**Paper Claim**:
"We investigate combining corruption-refinement learning with efficient attention mechanisms for jazz generation. While ProbSparse Attention proves incompatible, our deep decoder architecture achieves 75% recognition. Distilled model reaches 60% at 130ms latency, enabling interactive applications."

**Impact**: Weak accept or borderline at ISMIR 2026

---

### Scenario D: Failure Case (5% probability)

**Results**:
- Teacher: <70% jazz recognition (ProbSparse degrades too much)
- Student: <55% jazz recognition (distillation fails)
- Latency >150ms (not real-time)

**Action**: Pivot or defer to 2027
- Option 1: Submit to workshop instead of main conference
- Option 2: Focus on ablation study paper ("When does ProbSparse fail?")
- Option 3: Extend timeline to Month 26-27, try more solutions

---

## 10. Summary: What We Actually Know vs Hypothesize

### ✅ What We KNOW (Proven):
1. Music Informer achieves 21.73% speedup (proven in their paper)
2. ImprovNet achieves 79% jazz recognition (proven in their paper)
3. Magenta RT achieves <50ms inference latency (proven in their paper)
4. Corruption-refinement works for jazz (proven by ImprovNet)
5. Knowledge distillation works for NLP (DistilBERT: 97% retention)

### ⚠️ What We HYPOTHESIZE (To be validated):
1. **ProbSparse preserves jazz quality** (unproven - designed for time-series)
2. **ProbSparse + Corruption are compatible** (unproven - potential conflict)
3. **Hybrid architecture matches ImprovNet** (unproven - new combination)
4. **Jazz knowledge distills at 75-85%** (unproven - music ≠ NLP)
5. **<100ms end-to-end latency achievable** (unproven - system overhead unknown)

### 🎯 Research Contribution (If Successful):

**Primary Contribution**:
Empirical validation of whether:
1. ProbSparse Attention works for music (especially jazz)
2. Corruption-refinement scales via distillation
3. Real-time high-quality jazz generation is feasible

**Secondary Contribution**:
- Comprehensive ablation studies
- Fair baseline comparisons
- Open-source implementation for reproducibility

**Tertiary Contribution** (Best case):
- First real-time jazz system >65% recognition
- Hybrid architecture combining three SOTA techniques

---

## 11. Conclusion

This proposal outlines a **high-risk, high-reward** research project combining:
- Music Informer's efficiency (ProbSparse)
- ImprovNet's quality (corruption-refinement)
- Magenta RT's speed (streaming)

**Success is NOT guaranteed.** Multiple technical risks exist:
- ProbSparse may not work for music
- Corruption + ProbSparse may conflict
- Distillation may lose too much quality
- Real-time latency may be unachievable

**However**: Even partial success yields valuable contributions:
- Ablation studies reveal when ProbSparse works/fails
- Fair baseline comparisons advance the field
- Negative results ("ProbSparse doesn't work for jazz") are publishable

**Recommendation**:
✅ **Proceed with caution**
- Month 20: Validate ProbSparse early (GO/NO-GO decision)
- Month 23: Validate distillation (adjust expectations if needed)
- Month 25: Buffer for pivots and contingencies

**If successful → Strong ISMIR 2026 paper**
**If partial success → Acceptable ISMIR 2026 paper**
**If failure → Workshop paper or ISMIR 2027**

---

**This is science. We hypothesize, test, and report honestly—whether results match expectations or not.** 🔬

---

**End of Revised JazzFlow-RT Integration Strategy**
