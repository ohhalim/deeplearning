# 🎯 2026년 재즈 SOTA 모델 논문 제출 로드맵

> **목표**: 2026년 11월 ISMIR 또는 2026년 12월 NeurIPS에 "Real-Time Jazz Improvisation Model" 논문 제출

**시작일**: 2025년 11월
**논문 제출**: 2026년 11월 (ISMIR) 또는 2026년 12월 (NeurIPS)
**총 기간**: 24개월

---

## 📅 전체 타임라인 요약

| 기간 | 단계 | 목표 | 결과물 |
|------|------|------|--------|
| Month 1-2 | 기초 다지기 | Music Transformer 구현 | 작동하는 코드 |
| Month 3-4 | 오디오 기초 | VAE + WaveNet | 오디오 생성 이해 |
| Month 5-6 | Diffusion 기초 | DDPM 구현 | Diffusion 모델 완성 |
| Month 7-9 | 최신 모델 1 | MusicGen 분석 | EnCodec 이해 |
| Month 10-12 | 최신 모델 2 | Music Informer | MIDI 효율화 |
| Month 13-14 | 제어 기술 | Rule-Guided Diffusion | 조건부 생성 |
| Month 15-16 | 재즈 특화 | ImprovNet 분석 | 재즈 이해 |
| Month 17-18 | 실시간 생성 | Magenta RealTime | Fine-tuning |
| Month 19-20 | 연구 설계 | 재즈 데이터 수집 | Dataset 구축 |
| Month 21-22 | 모델 개발 | 자체 모델 훈련 | SOTA 모델 |
| Month 23 | 실험 & 평가 | 성능 검증 | 결과 분석 |
| Month 24 | 논문 작성 | 논문 완성 | 제출! |

---

## 📚 Month 1-2: Music Transformer 기초

### 학습 목표
- Transformer 아키텍처 완벽 이해
- MIDI 데이터 처리
- 음악 생성의 기본 파이프라인

### 이론 학습
- [ ] "Attention Is All You Need" 논문 정독 (3회 이상)
- [ ] "Music Transformer" 논문 정독
- [ ] Positional Encoding의 음악적 의미 이해
- [ ] Relative Attention 메커니즘

### 실습 과제
- [ ] Vanilla Transformer from scratch 구현
- [ ] MIDI 데이터 전처리 파이프라인
- [ ] Music Transformer 구현 (simple version)
- [ ] MAESTRO 데이터셋으로 학습 (소규모)

### 평가 기준
```python
# 성공 기준
1. 8-bar 멜로디 생성 가능
2. 기본적인 음악적 구조 (반복, 패턴)
3. Training loss < 1.5
4. Perplexity < 10
```

### 참고 자료
- 원본 구현: https://github.com/jason9693/MusicTransformer-pytorch
- MAESTRO 데이터셋: https://magenta.tensorflow.org/datasets/maestro

---

## 🎼 Month 3-4: VAE + WaveNet 오디오 생성 기초

### 학습 목표
- 오디오 신호 처리 이해
- Latent space 개념
- Raw audio waveform 생성

### 이론 학습
- [ ] "Auto-Encoding Variational Bayes" 논문
- [ ] "WaveNet: A Generative Model for Raw Audio" 논문
- [ ] Reparameterization trick 이해
- [ ] Dilated convolution 원리

### 실습 과제
- [ ] VAE 구현 (MNIST 먼저, 그 다음 오디오)
- [ ] 멜 스펙트로그램 변환 코드
- [ ] WaveNet 간단 버전 구현
- [ ] 짧은 오디오 샘플 (1-2초) 생성

### 평가 기준
```python
# 성공 기준
1. VAE로 의미있는 latent space 생성
2. Reconstruction loss < 0.1
3. WaveNet으로 들을 만한 소리 생성
4. 1초 오디오를 10초 내에 생성
```

### 데이터셋
- NSynth Dataset (악기 소리)
- FMA (Free Music Archive) 일부

---

## 🌊 Month 5-6: DDPM Diffusion 기초

### 학습 목표
- Diffusion 모델 완벽 이해
- Noise schedule 설계
- Denoising process 구현

### 이론 학습
- [ ] "Denoising Diffusion Probabilistic Models" 논문 정독
- [ ] "Improved DDPM" 논문
- [ ] Forward/Reverse process 수학적 이해
- [ ] Variance schedule 최적화

### 실습 과제
- [ ] DDPM from scratch (이미지 먼저)
- [ ] 오디오용 DDPM 구현
- [ ] Diffusion으로 멜 스펙트로그램 생성
- [ ] Conditional diffusion (클래스 조건부)

### 평가 기준
```python
# 성공 기준
1. CIFAR-10에서 FID < 20
2. 오디오 스펙트로그램 생성 성공
3. 1000 steps → 50 steps DDIM 가속
4. Conditional 생성 정확도 > 80%
```

### 코드 참고
- Denoising Diffusion PyTorch: https://github.com/lucidrains/denoising-diffusion-pytorch

---

## 🎵 Month 7-9: MusicGen 분석 및 재현

### 학습 목표
- 텍스트-투-음악 파이프라인
- EnCodec 오디오 압축 이해
- Autoregressive transformer for audio

### 이론 학습
- [ ] "Simple and Controllable Music Generation" (MusicGen 논문)
- [ ] "High Fidelity Neural Audio Compression" (EnCodec 논문)
- [ ] Residual Vector Quantization (RVQ)
- [ ] Classifier-Free Guidance

### 실습 과제
- [ ] MusicGen 코드 완전 분석
- [ ] EnCodec 동작 원리 이해 및 실험
- [ ] 작은 규모 MusicGen 재현 (100M 파라미터)
- [ ] 텍스트 조건부 생성 실험

### 평가 기준
```python
# 성공 기준
1. 30초 음악 생성 가능
2. 텍스트 프롬프트와 일치도 > 70%
3. FAD (Frechet Audio Distance) < 5.0
4. 자체 데이터셋으로 학습 성공
```

### 실험
- 다양한 프롬프트 테스트
- 멜로디 컨디셔닝 실험
- 재즈 데이터로 fine-tuning 시도

---

## 🎹 Month 10-12: Music Informer 구현

### 학습 목표
- ProbSparse attention 이해
- MIDI 생성 효율화
- Long sequence 처리

### 이론 학습
- [ ] "Music Informer" Nature 논문 정독
- [ ] Informer 원본 논문 (시계열)
- [ ] ProbSparse self-attention 수학
- [ ] Relative local attention

### 실습 과제
- [ ] Informer 기본 구현
- [ ] Music Informer 완전 재현
- [ ] MAESTRO 데이터셋으로 학습
- [ ] Music Transformer와 성능 비교

### 평가 기준
```python
# 성공 기준
1. 논문의 21.73% 연산 절감 재현
2. Music Transformer보다 빠른 학습
3. 음악적 품질 동등 이상
4. 512-bar 긴 시퀀스 생성 가능
```

### 연구 노트
- 왜 재즈에 효율적인가?
- 즉흥연주에 적용 가능성
- Real-time 변환 가능성

---

## 🎨 Month 13-14: Rule-Guided Diffusion

### 학습 목표
- 음악 이론 규칙 통합
- Non-differentiable guidance
- Stochastic Control Guidance (SCG)

### 이론 학습
- [ ] "Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion" (ICML 2024 Oral)
- [ ] Classifier-Free Guidance vs SCG
- [ ] Plug-and-play guidance 방법론

### 실습 과제
- [ ] Rule-Guided Diffusion 코드 분석
- [ ] 재즈 화성 규칙 구현 (ii-V-I progression 등)
- [ ] SCG로 코드 진행 제어
- [ ] 커스텀 규칙 함수 작성

### 평가 기준
```python
# 성공 기준
1. 화성 규칙 준수율 > 90%
2. 재즈 코드 진행 정확도 > 85%
3. 음악적 품질 유지 (rule 적용 후에도)
4. 다양한 재즈 스타일 규칙 적용 가능
```

### 재즈 규칙 예시
```python
# 구현할 규칙들
1. ii-V-I progression
2. Bebop scale patterns
3. Swing rhythm
4. Blue note usage
5. Chord extensions (7th, 9th, 11th, 13th)
```

---

## 🎺 Month 15-16: ImprovNet 재즈 특화

### 학습 목표
- 재즈 즉흥연주 메커니즘
- Style transfer 기법
- Corruption-refinement learning

### 이론 학습
- [ ] "ImprovNet" 논문 (arXiv 2502.04522) 정독
- [ ] Jazz improvisation theory
- [ ] Cross-genre style transfer
- [ ] Iterative refinement 방법론

### 실습 과제
- [ ] ImprovNet 코드 완전 분석 (GitHub)
- [ ] 클래식→재즈 변환 실험
- [ ] 9단계 스타일 제어 테스트
- [ ] 자체 재즈 데이터로 재학습

### 평가 기준
```python
# 성공 기준
1. 재즈 스타일 식별 > 79% (논문 기준)
2. 다양한 재즈 스타일 생성 (bebop, swing, cool)
3. 원곡 구조 유지하면서 재즈화
4. Chord-aware improvisation
```

### 재즈 데이터셋
- Weimar Jazz Database
- iRealPro charts
- PiJAMA dataset (200시간)
- 직접 수집한 재즈 MIDI

---

## ⚡ Month 17-18: Magenta RealTime 실시간 생성

### 학습 목표
- 실시간 생성 기법
- Streaming inference
- Fine-tuning for jazz

### 이론 학습
- [ ] Magenta RealTime 논문/기술문서
- [ ] SpectroStream codec 이해
- [ ] Real-time factor (RTF) 최적화
- [ ] Style embedding manipulation

### 실습 과제
- [ ] Magenta RT 설치 및 실행
- [ ] 재즈 데이터셋으로 fine-tuning
- [ ] Real-time streaming 구현
- [ ] Interactive control 시스템

### 평가 기준
```python
# 성공 기준
1. RTF > 1.0 (실시간보다 빠름)
2. 재즈 fine-tuning 후 스타일 유지
3. 10초 latency 이하로 스타일 전환
4. 48kHz stereo 품질 유지
```

### Fine-tuning 전략
```python
# 재즈 특화 학습
1. PiJAMA 200h + 자체 수집 300h
2. Frozen backbone + 재즈 style head
3. LoRA fine-tuning (효율적)
4. Continual learning (catastrophic forgetting 방지)
```

---

## 🚀 Month 19-20: 연구 설계 & 데이터 수집

### 연구 질문 정의

**핵심 질문**:
> "어떻게 실시간으로 재즈 이론을 준수하면서 고품질 즉흥연주를 생성할 것인가?"

### 제안 모델: **JazzFlow-RT**

**핵심 아이디어**:
```python
JazzFlow-RT = Magenta RealTime (실시간)
            + Rule-Guided Diffusion (재즈 이론)
            + ImprovNet (스타일 제어)
```

### 혁신 포인트
1. **Real-time rule-guided generation** (기존 연구 없음!)
2. **Jazz theory compliance in streaming** (새로운 접근)
3. **Interactive style morphing** (라이브 퍼포먼스)

### 데이터 수집 목표

**목표**: 500-1000시간 고품질 재즈 데이터

| 소스 | 시간 | 품질 | 형식 |
|------|------|------|------|
| PiJAMA | 200h | 높음 | MIDI + Audio |
| Weimar Jazz DB | 50h | 높음 | MIDI |
| YouTube (크롤링) | 300h | 중상 | Audio |
| iRealPro charts | 100h | 높음 | MIDI (생성) |
| 직접 연주 녹음 | 50h | 최고 | Audio |
| **Total** | **700h** | - | - |

### 데이터 전처리 파이프라인

```python
# 파이프라인
1. Audio → 48kHz stereo 변환
2. Source separation (Demucs) → 악기별 분리
3. Transcription (Spotify Basic Pitch) → MIDI
4. Chord detection (Chordino) → 화성 정보
5. Beat tracking → 리듬 정보
6. 품질 필터링 (SNR > 20dB)
7. 데이터 증강 (pitch shift, time stretch)
```

### 연구 설계

**실험 설정**:
```python
# Baseline 모델들
1. Magenta RT (vanilla)
2. MusicGen (fine-tuned)
3. ImprovNet
4. Rule-Guided Diffusion

# 제안 모델
5. JazzFlow-RT (우리 모델)

# 평가 메트릭
객관적:
- FAD (Frechet Audio Distance)
- Jazz theory compliance score
- RTF (Real-time Factor)
- Chord accuracy

주관적:
- MOS (Mean Opinion Score)
- Jazz musician evaluation (5-10명)
- ABX test (vs human performance)
```

---

## 🔬 Month 21-22: 모델 개발 & 훈련

### 아키텍처 설계

```python
class JazzFlowRT(nn.Module):
    """
    실시간 재즈 즉흥연주 생성 모델

    Components:
    1. MagentaRT backbone (800M params)
    2. Jazz-specific style encoder
    3. Real-time rule guidance module
    4. Streaming inference engine
    """

    def __init__(self):
        # 1. Backbone (pre-trained Magenta RT)
        self.backbone = MagentaRTBackbone.from_pretrained()

        # 2. Jazz Style Encoder (학습 가능)
        self.jazz_encoder = JazzStyleEncoder(
            embedding_dim=512,
            num_styles=10  # bebop, swing, cool, etc.
        )

        # 3. Rule Guidance Module (실시간 제약)
        self.rule_guide = RealTimeRuleGuide(
            rules=['chord_progression', 'rhythm', 'scale']
        )

        # 4. Streaming Generator
        self.stream_gen = StreamingGenerator(
            chunk_size=2.0,  # 2초 청크
            overlap=0.5      # 0.5초 오버랩
        )
```

### 훈련 전략

**Phase 1: Jazz Fine-tuning (2주)**
```python
# Magenta RT → Jazz 특화
- 데이터: 700시간 재즈
- 방법: LoRA fine-tuning
- Learning rate: 1e-5
- Batch size: 32
- GPU: 4x A100 (또는 Colab Pro+)
```

**Phase 2: Rule Integration (3주)**
```python
# Rule-guided generation 통합
- Jazz theory rules 추가
- Chord-aware generation
- Real-time constraint optimization
- Latency < 100ms 유지
```

**Phase 3: End-to-End Training (3주)**
```python
# 전체 시스템 최적화
- Multi-task learning
- Style control 정교화
- Interactive response 개선
```

### 실험 로그 관리

```python
# Weights & Biases 설정
import wandb

wandb.init(
    project="jazzflow-rt",
    config={
        "model": "JazzFlow-RT",
        "dataset": "700h-jazz",
        "target_conference": "ISMIR 2026"
    }
)

# 추적할 메트릭
- training_loss
- fad_score
- jazz_theory_compliance
- real_time_factor
- chord_accuracy
```

---

## 📊 Month 23: 실험 & 평가

### 객관적 평가

**1. FAD (Frechet Audio Distance)**
```python
# 목표: FAD < 3.0 (현재 SOTA: ~4.5)
- 계산: 생성 음악 vs 실제 재즈
- 도구: frechet_audio_distance library
```

**2. Jazz Theory Compliance**
```python
# 자동 평가 시스템 구축
def evaluate_jazz_theory(generated_midi):
    scores = {
        'chord_progression': check_chord_progression(),
        'scale_adherence': check_scale_usage(),
        'rhythm_swing': detect_swing_feel(),
        'blue_notes': count_blue_notes(),
        'voice_leading': check_voice_leading()
    }
    return np.mean(list(scores.values()))

# 목표: > 85% compliance
```

**3. Real-time Performance**
```python
# Latency 측정
- Chunk generation time: < 1.25s (for 2s audio)
- RTF: > 1.6
- Total latency: < 200ms
```

### 주관적 평가

**1. MOS (Mean Opinion Score)**
```python
# 설문 대상: 50명
- 일반 청취자: 30명
- 재즈 뮤지션: 15명
- 음악 교수: 5명

# 평가 항목 (1-5 scale)
- Overall quality
- Jazz authenticity
- Improvisation creativity
- Technical proficiency

# 목표: MOS > 4.0
```

**2. ABX Test**
```python
# 블라인드 테스트
A: 실제 재즈 연주
B: JazzFlow-RT 생성
X: A 또는 B (맞히기)

# 목표: < 60% 정답률 (구분 어려움 = 성공)
```

### Ablation Study

```python
# 각 컴포넌트 기여도 분석
실험 1: Magenta RT only (baseline)
실험 2: + Jazz fine-tuning
실험 3: + Rule guidance
실험 4: + Style control (full model)

# 각 단계별 성능 향상 측정
```

### 실패 케이스 분석

```python
# 모델이 실패하는 경우 분석
1. 복잡한 코드 진행 (예: Coltrane changes)
2. 극단적 템포 (매우 빠른 bebop)
3. 특이한 리듬 (polyrhythm)
4. 모던 재즈 (atonal)

# 각 케이스별 원인 분석 및 개선 방향
```

---

## 📝 Month 24: 논문 작성 & 제출

### 논문 구조 (ISMIR 형식)

```markdown
# JazzFlow-RT: Real-Time Jazz Improvisation Generation with Rule-Guided Streaming

## Abstract (200 words)
- 문제 정의
- 제안 방법
- 주요 결과
- 기여도

## 1. Introduction
1.1 Motivation (재즈 실시간 생성의 필요성)
1.2 Challenges (기존 방법의 한계)
1.3 Contributions (3-4개 핵심 기여)

## 2. Related Work
2.1 Music Generation Models
2.2 Real-time Audio Synthesis
2.3 Jazz-specific Models
2.4 Rule-guided Generation

## 3. Method
3.1 Overall Architecture
3.2 Jazz Fine-tuning Strategy
3.3 Real-time Rule Guidance
3.4 Streaming Inference

## 4. Experiments
4.1 Experimental Setup
4.2 Datasets
4.3 Baselines
4.4 Evaluation Metrics

## 5. Results
5.1 Objective Evaluation
5.2 Subjective Evaluation
5.3 Ablation Study
5.4 Real-time Performance

## 6. Analysis
6.1 Qualitative Analysis
6.2 Case Studies
6.3 Limitations

## 7. Conclusion
7.1 Summary
7.2 Future Work

## References (40-60 papers)
```

### 주요 Figure/Table 준비

```python
# 필수 시각화
Figure 1: Overall architecture diagram
Figure 2: Training pipeline
Figure 3: Real-time inference flow
Figure 4: Jazz theory compliance over time
Figure 5: Latency analysis
Figure 6: Ablation study results

Table 1: Dataset statistics
Table 2: Comparison with baselines (objective)
Table 3: MOS scores (subjective)
Table 4: Jazz theory compliance breakdown
Table 5: Real-time performance metrics
```

### 작성 일정 (4주)

**Week 1: Draft**
```
- Abstract, Introduction
- Method section
- Figure 초안
```

**Week 2: Experiments & Results**
```
- Experiments section
- Results section
- 모든 Table/Figure 완성
```

**Week 3: Polish**
```
- Related Work 보강
- Analysis section
- Conclusion
- References 정리
```

**Week 4: Review & Submit**
```
- 공저자 리뷰
- 영문 교정
- 최종 검토
- 제출!
```

### 투고 타겟

**1순위: ISMIR 2026**
```
- 마감: 2026년 4월 (예상)
- 결과: 2026년 7월
- 학회: 2026년 11월
- 장점: 음악 특화, 높은 인정도
```

**2순위: NeurIPS 2026**
```
- 마감: 2026년 5월 (예상)
- 결과: 2026년 9월
- 학회: 2026년 12월
- 장점: Top-tier, 더 높은 impact
```

**3순위: ICML 2027** (백업)
```
- 마감: 2027년 1월
- 리젝 시 백업 플랜
```

### 동영상 데모 준비

```python
# Supplementary Material
1. Demo video (3분)
   - Live jazz improvisation
   - Real-time style control
   - Comparison with baselines

2. Audio samples (20개)
   - 다양한 스타일
   - Before/after fine-tuning
   - Interactive demo

3. Code release
   - GitHub repository
   - Pre-trained weights
   - Colab demo notebook
```

---

## 🎯 핵심 성공 지표

### 연구 품질
- [ ] 새로운 방법론 제안 (Real-time rule-guided jazz)
- [ ] SOTA 성능 달성 (FAD < 3.0)
- [ ] 실용성 증명 (RTF > 1.6)
- [ ] 재즈 전문가 인정 (MOS > 4.0)

### 논문 품질
- [ ] 명확한 기여도 (3개 이상)
- [ ] 충분한 실험 (5개 이상 baseline)
- [ ] 재현 가능성 (코드 공개)
- [ ] 시각적 완성도 (고품질 figure)

### 학회 기준
- [ ] ISMIR acceptance rate: ~30%
- [ ] NeurIPS acceptance rate: ~25%
- [ ] Oral presentation 목표!

---

## 💰 예산 & 리소스

### 컴퓨팅 리소스

```python
# 필요 GPU 시간 (24개월)
Month 1-6:   Colab Pro ($10/month) = $60
Month 7-18:  Lambda Labs A100 (100h) = $1,100/month × 12 = $13,200
Month 19-24: Lambda Labs 4x A100 (200h) = $4,400/month × 6 = $26,400

Total: ~$40,000

# 예산 절약 방안
1. Google Colab Pro+ ($50/month) - 초기 단계
2. University GPU cluster (무료) - 가능하면
3. Kaggle 무료 GPU (30h/week)
4. Cloud credits (GCP, AWS for Startups)
```

### 데이터셋 비용
```
- 대부분 무료 (오픈 데이터)
- YouTube 크롤링 (무료, 합법적 사용)
- iRealPro 구독: $15
```

### 기타
```
- 참고 도서: $200
- Overleaf Premium: $30 (논문 작성)
- 학회 등록비: $500-800 (채택 시)
```

---

## 📖 필수 논문 읽기 리스트

### Transformers (5편)
1. Attention Is All You Need (Vaswani et al., 2017)
2. BERT (Devlin et al., 2018)
3. Music Transformer (Huang et al., 2018)
4. Transformer-XL (Dai et al., 2019)
5. Perceiver (Jaegle et al., 2021)

### Diffusion Models (5편)
6. DDPM (Ho et al., 2020)
7. Improved DDPM (Nichol & Dhariwal, 2021)
8. Classifier-Free Guidance (Ho & Salimans, 2022)
9. DDIM (Song et al., 2020)
10. Stable Diffusion (Rombach et al., 2022)

### Music Generation (10편)
11. WaveNet (van den Oord et al., 2016)
12. Jukebox (Dhariwal et al., 2020)
13. MusicGen (Copet et al., 2023)
14. AudioLM (Borsos et al., 2023)
15. MusicLM (Agostinelli et al., 2023)
16. Magenta RealTime (Google, 2025)
17. Music Informer (Sun et al., 2025)
18. ImprovNet (Bhandari et al., 2025)
19. Rule-Guided Diffusion (Huang et al., 2024)
20. MIDI-GPT (Pasquier et al., 2025)

### Jazz-specific (3편)
21. BebopNet (Trieu & Keller, 2020)
22. JazzGAN (Trieu et al., 2019)
23. DeepJ (Liu et al., 2017)

### Audio Codec (3편)
24. EnCodec (Défossez et al., 2022)
25. SoundStream (Zeghidour et al., 2021)
26. DAC (Kumar et al., 2023)

**Total: 26편 필수**

---

## 🚨 위험 요소 & 대응 전략

### 위험 1: GPU 부족
```
대응:
- Colab Pro+ 우선 사용
- Lambda Labs spot instance (저렴)
- Model parallelism으로 작은 GPU 활용
- LoRA fine-tuning으로 메모리 절약
```

### 위험 2: 성능 미달
```
대응:
- Baseline 모델로 점진적 개선
- Ablation study로 병목 지점 파악
- 재즈 전문가 피드백 반영
- 목표 하향 조정 (SOTA → near-SOTA)
```

### 위험 3: 논문 리젝
```
대응:
- 조기에 프리프린트 (arXiv) 공개
- 워크샵 먼저 제출 (ISMIR WiMIR, NeurIPS workshop)
- 피드백 반영 후 재제출
- 2nd choice 학회 준비 (ICASSP, IJCAI)
```

### 위험 4: 시간 부족
```
대응:
- 타임라인 2주 버퍼 포함
- 중요도 낮은 실험 skip
- Minimal viable paper 우선
- 확장 버전은 journal 제출
```

---

## 📞 커뮤니티 & 멘토링

### 온라인 커뮤니티
```
- r/MachineLearning (Reddit)
- ISMIR Discussion Group
- Magenta Google Group
- Hugging Face Discord
```

### 찾아야 할 멘토
```
1. 재즈 뮤지션 (음악적 피드백)
2. ML 연구자 (기술적 조언)
3. 논문 작성 경험자 (writing)
```

### 오픈소스 기여
```
- Magenta 프로젝트 contribute
- 자신의 코드 GitHub 공개
- 블로그 포스팅으로 지식 공유
```

---

## 🎉 마일스톤 체크리스트

### Month 6 체크포인트
- [ ] Music Transformer 완성
- [ ] DDPM 구현 완료
- [ ] 첫 음악 생성 성공
- [ ] 기초 튼튼

### Month 12 체크포인트
- [ ] MusicGen 이해 완료
- [ ] Music Informer 재현
- [ ] 재즈 데이터 50시간 수집
- [ ] 중간 발표 준비

### Month 18 체크포인트
- [ ] Magenta RT fine-tuning 성공
- [ ] Rule-guided generation 작동
- [ ] 재즈 데이터 300시간 확보
- [ ] 논문 아이디어 확정

### Month 24 목표
- [ ] **논문 제출 완료**
- [ ] 코드 GitHub 공개
- [ ] Demo 사이트 오픈
- [ ] arXiv 프리프린트

---

## 📈 성장 추적

### 매월 작성할 것
```markdown
# Month X Progress Report

## 학습 내용
- 읽은 논문: [리스트]
- 구현한 코드: [링크]
- 새로 배운 개념: [설명]

## 어려웠던 점
- 문제: [상세]
- 해결: [방법]

## 다음 달 목표
- [ ] 목표 1
- [ ] 목표 2
- [ ] 목표 3

## 느낀 점
[자유롭게 작성]
```

---

## 🎯 최종 목표

```
2026년 11월, ISMIR 무대에서:

"JazzFlow-RT: Real-Time Jazz Improvisation Generation
with Rule-Guided Streaming"

발표자: [당신의 이름]

청중의 박수와 함께,
당신은 재즈 AI 연구의 새로운 장을 열었습니다.

이것이 목표입니다.
24개월 후, 우리 여기서 만납시다! 🎺🎉
```

---

**작성일**: 2025-11-16
**최종 수정**: 2025-11-16
**버전**: 1.0
