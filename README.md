# 🎺 재즈 음악 생성 AI - SOTA 모델 개발 로드맵

> **목표**: 2026년 11월 ISMIR 또는 NeurIPS에 "Real-Time Jazz Improvisation Model" 논문 제출

**시작일**: 2025년 11월
**총 기간**: 24개월
**최종 목표**: JazzFlow-RT - 실시간 재즈 즉흥연주 생성 SOTA 모델

---

## 📋 프로젝트 개요

이 저장소는 딥러닝 기초부터 최신 음악 생성 SOTA 모델까지 학습하고, 최종적으로 재즈 특화 모델을 개발하여 논문을 발표하는 것을 목표로 합니다.

### 핵심 질문

> "어떻게 실시간으로 재즈 이론을 준수하면서 고품질 즉흥연주를 생성할 것인가?"

### 제안 모델: JazzFlow-RT

```
JazzFlow-RT = Magenta RealTime (실시간 생성, RTF 1.6)
            + Rule-Guided Diffusion (재즈 이론 준수)
            + ImprovNet (스타일 제어)
```

**혁신 포인트**:
1. 🚀 Real-time rule-guided jazz generation (세계 최초!)
2. 🎵 Jazz theory compliance in streaming
3. 🎹 Interactive style morphing for live performance

---

## 📁 프로젝트 구조

```
deeplearning/
│
├── README.md (이 파일)
├── music-generation-learning-roadmap.md (전체 학습 로드맵)
├── 2026-paper-submission-roadmap.md (논문 제출 상세 계획)
│
├── month-02-music-transformer/         # Month 1-2: Transformer 기초
│   ├── 01_basic_transformer.py
│   ├── 02_midi_preprocessing.py
│   ├── 03_training.py
│   └── README.md
│
├── month-04-vae-wavenet/               # Month 3-4: 오디오 생성 기초
│   ├── 01_vae_audio.py
│   ├── 02_wavenet.py
│   └── README.md
│
├── month-06-ddpm/                      # Month 5-6: Diffusion Models
│   ├── 01_ddpm_implementation.py
│   ├── 02_ddim_fast_sampling.py
│   └── README.md
│
├── month-09-musicgen/                  # Month 7-9: MusicGen 분석
│   ├── 01_musicgen_analysis.py
│   ├── 02_encodec_codec.py
│   └── README.md
│
├── month-12-music-informer/            # Month 10-12: Music Informer
│   ├── 01_informer_implementation.py
│   ├── 02_probsparse_attention.py
│   └── README.md
│
├── month-14-rule-guided-diffusion/     # Month 13-14: Rule-Guided
│   ├── 01_rule_guidance.py
│   ├── 02_jazz_rules.py
│   └── README.md
│
├── month-16-improvnet/                 # Month 15-16: ImprovNet
│   ├── 01_improvnet_analysis.py
│   ├── 02_style_transfer.py
│   └── README.md
│
├── month-18-magenta-realtime/          # Month 17-18: Magenta RT
│   ├── 01_finetune_jazz.py
│   ├── 02_realtime_inference.py
│   └── README.md
│
└── month-20-24-jazz-sota/              # Month 19-24: 연구 & 논문
    ├── research_plan.md
    ├── 01_jazzflow_rt.py
    ├── 02_data_collection.py
    ├── 03_training.py
    ├── 04_evaluation.py
    ├── 05_paper/
    └── README.md
```

---

## 🗓️ 학습 타임라인

| 기간 | 단계 | 학습 내용 | 핵심 결과물 |
|------|------|---------|-----------|
| **Month 1-2** | 기초 | Music Transformer | 8-bar 멜로디 생성 |
| **Month 3-4** | 오디오 | VAE + WaveNet | 오디오 재구성 |
| **Month 5-6** | Diffusion | DDPM | 스펙트로그램 생성 |
| **Month 7-9** | 최신 모델 | MusicGen 분석 | 텍스트→음악 이해 |
| **Month 10-12** | 효율성 | Music Informer | MIDI 효율화 |
| **Month 13-14** | 제어 | Rule-Guided Diffusion | 재즈 이론 적용 |
| **Month 15-16** | 재즈 | ImprovNet | 스타일 변환 |
| **Month 17-18** | 실시간 | Magenta RealTime | Jazz fine-tuning |
| **Month 19-20** | 연구 | 데이터 수집 | 700h 재즈 DB |
| **Month 21-22** | 개발 | JazzFlow-RT 구현 | SOTA 모델 |
| **Month 23** | 실험 | 성능 평가 | 결과 분석 |
| **Month 24** | 논문 | 작성 & 제출 | **ISMIR/NeurIPS** |

---

## 🎯 핵심 마일스톤

### Month 6 체크포인트 ✓
- [ ] Music Transformer 완성
- [ ] DDPM 구현 완료
- [ ] 첫 음악 생성 성공

### Month 12 체크포인트 ✓
- [ ] MusicGen 이해 완료
- [ ] Music Informer 재현
- [ ] 재즈 데이터 50시간 수집

### Month 18 체크포인트 ✓
- [ ] Magenta RT fine-tuning 성공
- [ ] Rule-guided generation 작동
- [ ] 재즈 데이터 300시간 확보

### Month 24 최종 목표 🎉
- [ ] **논문 제출 완료**
- [ ] 코드 GitHub 공개
- [ ] Demo 사이트 오픈
- [ ] arXiv 프리프린트

---

## 📊 성공 지표

### Must Have (필수)

| 메트릭 | 목표 | 현재 SOTA |
|--------|------|----------|
| FAD | < 4.0 | ~4.5 |
| Jazz Theory Compliance | > 80% | N/A |
| Real-time Factor (RTF) | > 1.0 | 1.6 (Magenta RT) |
| 논문 제출 | ✓ | - |

### Nice to Have (목표)

| 메트릭 | 목표 | 의미 |
|--------|------|------|
| FAD | < 3.0 | **SOTA 달성!** |
| Jazz Theory Compliance | > 85% | 전문가 수준 |
| RTF | > 1.6 | Magenta RT 수준 |
| MOS | > 4.0 | 높은 주관적 품질 |
| ABX Test | < 60% | 인간과 구분 어려움 |

### Dream Goal (꿈)

- 🏆 ISMIR Best Paper Award
- 🎤 NeurIPS Spotlight/Oral
- 🎺 재즈 뮤지션들의 실제 사용
- 📈 후속 연구로 SOTA 경신

---

## 🛠️ 기술 스택

### 딥러닝 프레임워크
```bash
PyTorch >= 2.0
torchaudio
transformers (HuggingFace)
diffusers
```

### 오디오 처리
```bash
librosa
pretty_midi
music21
soundfile
```

### 학습 관리
```bash
wandb (Weights & Biases)
tensorboard
```

### 데이터 수집
```bash
yt-dlp (YouTube 크롤링)
demucs (Source separation)
basic-pitch (Transcription)
```

---

## 📚 필수 논문 읽기 (26편)

### Transformers (5편)
1. ✓ Attention Is All You Need (Vaswani et al., 2017)
2. ✓ Music Transformer (Huang et al., 2018)
3. BERT (Devlin et al., 2018)
4. GPT-2 (Radford et al., 2019)
5. Perceiver (Jaegle et al., 2021)

### Diffusion Models (5편)
6. ✓ DDPM (Ho et al., 2020)
7. Improved DDPM (Nichol & Dhariwal, 2021)
8. DDIM (Song et al., 2020)
9. Classifier-Free Guidance (Ho & Salimans, 2022)
10. Stable Diffusion (Rombach et al., 2022)

### Music Generation (10편)
11. WaveNet (van den Oord et al., 2016)
12. Jukebox (Dhariwal et al., 2020)
13. ✓ MusicGen (Copet et al., 2023)
14. AudioLM (Borsos et al., 2023)
15. MusicLM (Agostinelli et al., 2023)
16. ✓ Magenta RealTime (Google, 2025)
17. ✓ Music Informer (Sun et al., 2025)
18. ✓ ImprovNet (Bhandari et al., 2025)
19. ✓ Rule-Guided Diffusion (Huang et al., 2024)
20. ✓ MIDI-GPT (Pasquier et al., 2025)

### Jazz & Audio (6편)
21. BebopNet (Trieu & Keller, 2020)
22. DeepJ (Liu et al., 2017)
23. EnCodec (Défossez et al., 2022)
24. SoundStream (Zeghidour et al., 2021)
25. Lyria (Google DeepMind, 2024)
26. Stable Audio 2.0 (Stability AI, 2024)

---

## 💻 시작하기

### 1. 환경 설정

```bash
# Python 3.10 권장
conda create -n music-ai python=3.10
conda activate music-ai

# 기본 라이브러리 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers wandb
pip install librosa pretty_midi music21 soundfile
pip install tqdm matplotlib seaborn
```

### 2. Month 2부터 시작

```bash
cd month-02-music-transformer
python 01_basic_transformer.py

# MAESTRO 데이터셋 다운로드
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip

# 학습 시작
python 03_training.py
```

### 3. 진행 상황 추적

각 월별 폴더의 `README.md`를 확인하고 체크리스트를 완료하세요.

---

## 📖 학습 자료

### 온라인 강의 (무료)
- [Fast.ai Deep Learning](https://course.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### 추천 도서
- "Deep Learning" - Ian Goodfellow
- "Hands-On Machine Learning" - Aurélien Géron
- "Speech and Language Processing" - Jurafsky & Martin

### 커뮤니티
- r/MachineLearning (Reddit)
- ISMIR Discussion Group
- Hugging Face Discord
- Papers With Code

---

## 💰 예산 & 리소스

### GPU 비용 (24개월)

```
Month 1-6:   Colab Pro ($10/month) × 6 = $60
Month 7-18:  Lambda A100 (100h/month) × 12 = $13,200
Month 19-24: Lambda 4xA100 (200h/month) × 6 = $26,400

Total: ~$40,000

절약 방안:
✓ Google Colab Pro+ ($50/month)
✓ Kaggle 무료 GPU (30h/week)
✓ University GPU cluster (무료)
✓ Cloud credits (GCP $300)
```

### 기타 비용
```
참고 도서: $200
학회 등록비: $500-800 (채택 시)
데이터 수집: $100
Total: ~$1,000
```

---

## 🚨 위험 관리

| 위험 | 확률 | 대응 전략 |
|------|------|---------|
| GPU 부족 | 중 | Colab Pro+, 모델 크기 축소, LoRA |
| 성능 미달 | 중 | 점진적 개선, near-SOTA도 OK |
| 논문 리젝 | 중 | arXiv 먼저, workshop, 재제출 |
| 시간 부족 | 고 | 우선순위, 최소 실험, 버퍼 2주 |
| 데이터 부족 | 저 | 데이터 증강, synthetic data |

---

## 📈 진행 상황 추적

매월 말에 `progress_reports/month-XX.md` 파일을 작성하세요:

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

## 🎉 최종 목표

```
2026년 11월, ISMIR 무대에서:

"JazzFlow-RT: Real-Time Jazz Improvisation Generation
 with Rule-Guided Streaming"

발표자: [당신의 이름]

청중의 박수와 함께,
당신은 재즈 AI 연구의 새로운 장을 열었습니다.

24개월 후, 우리 여기서 만납시다! 🎺🎉
```

---

## 📞 연락 & 기여

- 질문/이슈: GitHub Issues
- 진행 상황 공유: Discussions
- 코드 개선: Pull Requests 환영!

---

## 📄 라이선스

MIT License - 학습 및 연구 목적으로 자유롭게 사용 가능

---

**작성일**: 2025-11-16
**최종 수정**: 2025-11-16
**버전**: 1.0
**상태**: 🚀 진행 중

**Let's make SOTA jazz AI together! 🎺🤖**
