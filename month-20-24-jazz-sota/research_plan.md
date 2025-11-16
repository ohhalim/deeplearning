# Month 20-24: JazzFlow-RT 개발 및 논문 작성

## 연구 목표

**핵심 질문**: 어떻게 실시간으로 재즈 이론을 준수하면서 고품질 즉흥연주를 생성할 것인가?

## 제안 모델: JazzFlow-RT

### 핵심 아이디어

```
JazzFlow-RT = Magenta RealTime (실시간 생성)
            + Rule-Guided Diffusion (재즈 이론 준수)
            + ImprovNet (스타일 제어)
```

### 혁신 포인트

1. **Real-time rule-guided generation** (기존 연구 없음!)
2. **Jazz theory compliance in streaming** (새로운 접근)
3. **Interactive style morphing** (라이브 퍼포먼스 가능)

---

## Month 20: 데이터 수집 & 전처리

### 데이터 수집 목표: 700시간

| 소스 | 시간 | 형식 | 수집 방법 |
|------|------|------|---------|
| PiJAMA | 200h | MIDI + Audio | GitHub 다운로드 |
| Weimar Jazz DB | 50h | MIDI | 공식 웹사이트 |
| YouTube | 300h | Audio | yt-dlp 크롤링 |
| iRealPro charts | 100h | MIDI | iReal Pro 앱 |
| 직접 녹음 | 50h | Audio | 스튜디오 녹음 |

### 체크리스트

- [ ] PiJAMA 데이터셋 다운로드
- [ ] YouTube 재즈 크롤링 스크립트 작성
- [ ] iRealPro → MIDI 변환 파이프라인
- [ ] 오디오 품질 필터링 (SNR > 20dB)
- [ ] 메타데이터 정리 (스타일, BPM, 키)

### 전처리 파이프라인

```python
# pipeline.py
1. Audio → 48kHz stereo 변환
2. Source separation (Demucs) → 악기별 분리
3. Transcription (Basic Pitch) → MIDI
4. Chord detection (Chordino) → 화성 정보
5. Beat tracking → 리듬 정보
6. 데이터 증강 (pitch shift, time stretch)
7. Train/Val/Test split (80/10/10)
```

---

## Month 21: 모델 개발

### 아키텍처 설계

```python
class JazzFlowRT(nn.Module):
    """
    실시간 재즈 즉흥연주 생성 모델
    """

    def __init__(self):
        # 1. Backbone: Magenta RT (800M params, frozen)
        self.backbone = MagentaRTBackbone.from_pretrained()

        # 2. Jazz Style Encoder (학습 가능)
        self.jazz_encoder = JazzStyleEncoder(
            embedding_dim=512,
            num_styles=10  # bebop, swing, cool, modal, etc.
        )

        # 3. Rule Guidance Module (실시간)
        self.rule_guide = RealTimeRuleGuide(
            rules=[
                'chord_progression',  # ii-V-I 등
                'rhythm',             # swing feel
                'scale',              # bebop scale
                'blue_notes',         # 특징적 음정
                'voice_leading'       # 목소리 진행
            ]
        )

        # 4. Streaming Generator
        self.stream_gen = StreamingGenerator(
            chunk_size=2.0,   # 2초 청크
            overlap=0.5,      # 0.5초 오버랩
            buffer_size=10.0  # 10초 버퍼
        )
```

### Phase 1: Jazz Fine-tuning (2주)

```bash
# LoRA fine-tuning on 700h jazz
python train_jazz_finetuning.py \
    --data_dir /path/to/jazz/700h \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --lora_rank 16 \
    --num_epochs 10
```

### Phase 2: Rule Integration (3주)

```bash
# Integrate jazz theory rules
python train_rule_guided.py \
    --pretrained checkpoints/jazz_finetuned.pt \
    --rules chord+rhythm+scale \
    --lambda_rule 0.1
```

### Phase 3: End-to-End (3주)

```bash
# Full system optimization
python train_e2e.py \
    --model jazzflow_rt \
    --optimize latency+quality+compliance
```

---

## Month 22: 실험 & 평가

### 객관적 평가

#### 1. FAD (Fréchet Audio Distance)
```python
# evaluate_fad.py
from frechet_audio_distance import FrechetAudioDistance

fad = FrechetAudioDistance()
score = fad.score(generated_dir, real_jazz_dir)
print(f"FAD: {score:.2f}")  # 목표: < 3.0
```

#### 2. Jazz Theory Compliance
```python
# evaluate_jazz_theory.py
def jazz_theory_score(midi_path):
    scores = {
        'chord_progression': check_chord_progression(midi_path),
        'scale_adherence': check_scale_usage(midi_path),
        'swing_ratio': detect_swing_feel(midi_path),
        'blue_notes': count_blue_notes(midi_path),
        'voice_leading': check_voice_leading(midi_path)
    }
    return np.mean(list(scores.values()))

# 목표: > 85%
```

#### 3. Real-time Performance
```python
# benchmark_latency.py
import time

start = time.time()
audio = model.generate(duration=2.0)
elapsed = time.time() - start

rtf = elapsed / 2.0
print(f"RTF: {rtf:.2f}")  # 목표: > 1.6
```

### 주관적 평가

#### MOS (Mean Opinion Score) 설문

```markdown
대상: 50명
- 일반 청취자: 30명
- 재즈 뮤지션: 15명
- 음악 교수: 5명

질문 (1-5 scale):
1. Overall quality (전체 품질)
2. Jazz authenticity (재즈다움)
3. Improvisation creativity (즉흥성)
4. Technical proficiency (기술적 완성도)

목표: MOS > 4.0
```

#### ABX 블라인드 테스트

```
A: 실제 재즈 연주 (인간)
B: JazzFlow-RT 생성
X: A 또는 B (맞히기)

목표: < 60% 정답률 (= 구분하기 어려움)
```

### Ablation Study

```python
실험 1: Magenta RT only (baseline)
실험 2: + Jazz fine-tuning
실험 3: + Rule guidance
실험 4: + Style control (full JazzFlow-RT)

각 단계별 성능 향상 측정
```

---

## Month 23: 논문 작성 (Draft)

### 논문 구조 (ISMIR 형식)

```markdown
Title: JazzFlow-RT: Real-Time Jazz Improvisation Generation
       with Rule-Guided Streaming

Abstract (200 words)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We present JazzFlow-RT, the first real-time jazz
improvisation generation system that maintains both
high fidelity and jazz theory compliance...

1. Introduction
   1.1 Motivation
   1.2 Challenges
   1.3 Contributions

2. Related Work
   2.1 Music Generation Models
   2.2 Real-time Audio Synthesis
   2.3 Jazz-specific Approaches
   2.4 Rule-guided Generation

3. Method
   3.1 Overall Architecture
   3.2 Jazz Fine-tuning Strategy
   3.3 Real-time Rule Guidance
   3.4 Streaming Inference

4. Experiments
   4.1 Experimental Setup
   4.2 Datasets (700h jazz)
   4.3 Baselines
   4.4 Evaluation Metrics

5. Results
   5.1 Objective Evaluation
       - FAD: 2.8 (SOTA!)
       - Theory compliance: 87%
       - RTF: 1.7
   5.2 Subjective Evaluation
       - MOS: 4.2
       - ABX: 57% (barely distinguishable!)
   5.3 Ablation Study
   5.4 Real-time Performance

6. Analysis & Discussion
   6.1 Qualitative Analysis
   6.2 Case Studies
   6.3 Limitations
   6.4 Future Work

7. Conclusion

References (50+ papers)
```

### 주요 Figure/Table

```python
Figure 1: JazzFlow-RT Architecture
Figure 2: Training Pipeline
Figure 3: Real-time Inference Flow
Figure 4: Jazz Theory Compliance Over Time
Figure 5: Latency Analysis
Figure 6: Ablation Study Results
Figure 7: MOS Scores by Listener Group

Table 1: Dataset Statistics (700h)
Table 2: Comparison with Baselines
Table 3: Objective Metrics
Table 4: Subjective Evaluation (MOS)
Table 5: Jazz Theory Breakdown
Table 6: Real-time Performance
```

---

## Month 24: 논문 완성 & 제출

### Week 1-2: 리뷰 & 수정

- [ ] 공저자 리뷰 (만약 있다면)
- [ ] 영문 교정 (Grammarly, DeepL Write)
- [ ] Figure 고해상도 제작
- [ ] 참고문헌 정리 (BibTeX)

### Week 3: Supplementary Material

```
1. Demo Video (3분)
   - Live jazz improvisation showcase
   - Real-time style control demo
   - Comparison with baselines

2. Audio Samples (20개)
   - 다양한 재즈 스타일
   - Before/After fine-tuning
   - Interactive morphing

3. Code & Weights
   - GitHub repository 공개
   - Pre-trained weights (HuggingFace)
   - Colab demo notebook
   - Docker container
```

### Week 4: 제출!

#### 1순위: ISMIR 2026

```
마감: 2026년 4월 (예상)
결과: 2026년 7월
학회: 2026년 11월 (어딘가 유럽)

장점:
- 음악 정보 검색 최고 학회
- Acceptance rate ~30%
- 재즈 연구 환영받음
```

#### 2순위: NeurIPS 2026

```
마감: 2026년 5월 (예상)
결과: 2026년 9월
학회: 2026년 12월

장점:
- Top-tier ML 학회
- Impact factor 높음
- Oral 목표!
```

#### Backup: ICML 2027 or IEEE TASLP

---

## 성공 지표

### Must Have (필수)

- [ ] FAD < 4.0
- [ ] Jazz theory compliance > 80%
- [ ] RTF > 1.0 (실시간)
- [ ] 논문 제출 완료

### Nice to Have (목표)

- [ ] FAD < 3.0 (SOTA)
- [ ] Jazz theory compliance > 85%
- [ ] RTF > 1.6 (Magenta RT 수준)
- [ ] MOS > 4.0
- [ ] ABX < 60%
- [ ] Oral presentation

### Dream Goal (꿈)

- [ ] ISMIR Best Paper Award
- [ ] NeurIPS Spotlight/Oral
- [ ] 재즈 뮤지션들의 실제 사용
- [ ] 후속 연구 (SOTA 경신!)

---

## 예산 & 리소스

### GPU 비용 (6개월)

```
Month 19-20: A100 100h/month × 2 = $2,200
Month 21-22: 4x A100 200h/month × 2 = $8,800
Month 23-24: A100 50h/month × 2 = $1,100

Total: ~$12,000

절약 방안:
- Lambda Labs spot instance
- Google Colab Pro+ ($50/month)
- University GPU cluster (무료)
```

### 기타

```
- 학회 등록비: $500-800
- Audio 장비 (녹음): $500
- 설문 참가자 사례비: $500
- Overleaf Premium: $30
```

---

## 위험 관리

### 위험 1: 성능 미달

**대응**:
- Baseline 모델(Magenta RT)도 충분히 좋음
- 점진적 개선 보여주기
- "near-SOTA"도 충분한 기여도

### 위험 2: 데이터 부족

**대응**:
- 데이터 증강 (pitch shift, tempo)
- Synthetic 데이터 생성
- Transfer learning 강화

### 위험 3: 실시간 속도 부족

**대응**:
- Model distillation
- Quantization (INT8)
- Chunk size 조정

### 위험 4: 논문 리젝

**대응**:
- arXiv 프리프린트 먼저
- Workshop 제출 (ISMIR WiMIR)
- 피드백 반영 후 재제출
- 2nd choice: ICASSP, IJCAI

---

## 타임라인 체크리스트

### Month 20
- [ ] 재즈 데이터 300h 수집
- [ ] 전처리 파이프라인 완성
- [ ] 데이터셋 통계 분석

### Month 21
- [ ] JazzFlow-RT 구현 완료
- [ ] Jazz fine-tuning 완료
- [ ] Rule guidance 통합

### Month 22
- [ ] 모든 실험 완료
- [ ] Baseline 비교 완료
- [ ] 평가 완료 (객관+주관)

### Month 23
- [ ] 논문 초안 완성
- [ ] 모든 Figure/Table 완성
- [ ] Demo video 제작

### Month 24
- [ ] 논문 최종 제출
- [ ] 코드 공개
- [ ] 보충 자료 업로드

---

## 2026년 11월, ISMIR에서...

```
당신은 무대에 섭니다.

"JazzFlow-RT: Real-Time Jazz Improvisation
 Generation with Rule-Guided Streaming"

청중의 눈이 빛납니다.
재즈 연주자들이 고개를 끄덕입니다.
리뷰어들이 미소 짓습니다.

당신은 해냈습니다.

재즈 AI 연구의 새로운 장을 열었습니다.

This is your goal.
Let's make it happen! 🎺🎉
```

---

**작성일**: 2025-11-16
**다음 업데이트**: Month 20 시작 시
