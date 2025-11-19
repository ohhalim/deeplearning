# 🎵 찰리 파커 AI 만들기: 12개월 완전 로드맵

**목표**: 0원부터 시작해서 HuggingFace에 올릴 수 있는 재즈 AI 모델 만들기

**대상**: 고졸, GPU 없음, 예산 월 5만원, 백엔드 개발자 지망생
**최종 결과물**: GitHub 포트폴리오 + HuggingFace 모델 + 웹 데모 + AI 엔지니어 취업

---

## 📅 전체 타임라인 (12개월)

| Phase | 기간 | 목표 | 예산 | 결과물 |
|-------|------|------|------|--------|
| **Phase 1** | Month 1 | 환경 세팅 + 데이터 수집 | $0 | MIDI 데이터셋 |
| **Phase 2** | Month 2 | Small Model 학습 | $30 | 첫 번째 모델 |
| **Phase 3** | Month 3 | Fine-tuning 실험 | $50 | HuggingFace 모델 |
| **Phase 4** | Month 4-6 | 웹 데모 + API | $150 | 웹 서비스 |
| **Phase 5** | Month 7-9 | 포트폴리오 확장 | $150 | 3개 모델 |
| **Phase 6** | Month 10-12 | AI 엔지니어 취업 | $0 | 취업 성공 |

**총 예산**: $380 (1년간, 월 평균 $32 = 약 4만원)

---

## 🎯 각 Phase별 상세 가이드

### Phase 1: 환경 세팅 + 데이터 수집 (Month 1)

**목표**:
- ✅ 무료로 개발 환경 세팅
- ✅ 찰리 파커 MIDI 데이터 50-100곡 수집
- ✅ 데이터 전처리 파이프라인 구축

**예산**: $0 (완전 무료)

**학습 자료**:
```
roadmap/phase1-setup/
├── README.md                    # Phase 1 전체 가이드
├── 01_colab_setup.md           # Google Colab 무료 사용법
├── 02_data_collection.md       # MIDI 데이터 수집 가이드
├── 03_midi_preprocessing.py    # MIDI → 토큰 변환 코드
├── 04_data_exploration.ipynb   # 데이터 탐색 노트북
└── requirements.txt            # 필요한 라이브러리
```

**체크리스트**:
- [ ] Google Colab 계정 만들기
- [ ] 찰리 파커 MIDI 50곡 다운로드
- [ ] MIDI 파일을 토큰으로 변환
- [ ] 데이터 통계 확인 (평균 길이, 음역대 등)

**→ [Phase 1 시작하기](./phase1-setup/README.md)**

---

### Phase 2: Small Model 학습 (Month 2)

**목표**:
- ✅ 작은 Transformer 모델 (10M params) 처음부터 학습
- ✅ 찰리 파커 스타일 생성 확인
- ✅ 첫 번째 GitHub 프로젝트 완성

**예산**: $30 (Runpod RTX 3090, ~30시간 학습)

**학습 자료**:
```
roadmap/phase2-small-model/
├── README.md                    # Phase 2 전체 가이드
├── tokenizer.py                # MIDI Tokenizer 클래스
├── model.py                    # Transformer 모델 정의
├── dataset.py                  # PyTorch Dataset
├── train.py                    # 학습 스크립트
├── generate.py                 # 음악 생성 스크립트
├── config.yaml                 # 하이퍼파라미터 설정
└── colab_train.ipynb          # Colab 학습 노트북
```

**체크리스트**:
- [ ] Runpod 계정 만들고 $10 충전
- [ ] Small Transformer 모델 학습 (10시간)
- [ ] 생성된 MIDI 파일 들어보기
- [ ] GitHub에 코드 + 모델 업로드

**→ [Phase 2 시작하기](./phase2-small-model/README.md)**

---

### Phase 3: Fine-tuning 실험 (Month 3)

**목표**:
- ✅ 기존 Music Transformer를 찰리 파커 데이터로 Fine-tuning
- ✅ HuggingFace에 모델 업로드
- ✅ Gradio 웹 데모 만들기

**예산**: $50 (Colab Pro + Runpod)

**학습 자료**:
```
roadmap/phase3-finetuning/
├── README.md                    # Phase 3 전체 가이드
├── download_pretrained.py      # Pretrained 모델 다운로드
├── finetune.py                 # Fine-tuning 스크립트
├── evaluate.py                 # 모델 평가 (perplexity 등)
├── gradio_demo.py              # Gradio 웹 데모
├── upload_to_hf.py             # HuggingFace 업로드
└── colab_finetune.ipynb       # Colab Fine-tuning
```

**체크리스트**:
- [ ] Google Magenta Music Transformer 다운로드
- [ ] 찰리 파커 데이터로 Fine-tuning (20시간)
- [ ] HuggingFace에 모델 업로드
- [ ] Gradio 데모 만들어서 HF Spaces에 배포

**→ [Phase 3 시작하기](./phase3-finetuning/README.md)**

---

### Phase 4: 웹 데모 + API 서버 (Month 4-6)

**목표**:
- ✅ Spring Boot API 서버 구축 (백엔드 스킬 활용)
- ✅ React 프론트엔드 (간단한 UI)
- ✅ 실시간 재즈 생성 웹 서비스

**예산**: $150 (모델 최적화 실험)

**학습 자료**:
```
roadmap/phase4-advanced/
├── README.md
├── backend/                    # Spring Boot API
│   ├── src/main/java/...
│   ├── pom.xml
│   └── README.md
├── ml_service/                 # Python ML 서비스 (FastAPI)
│   ├── app.py
│   ├── model_loader.py
│   └── requirements.txt
├── frontend/                   # React 프론트엔드
│   ├── src/
│   ├── package.json
│   └── README.md
└── deployment/                 # Docker + 배포
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
```

**체크리스트**:
- [ ] FastAPI로 ML 추론 서버 구축
- [ ] Spring Boot에서 FastAPI 호출
- [ ] React로 간단한 UI 만들기
- [ ] 로컬에서 전체 시스템 테스트

**→ [Phase 4 시작하기](./phase4-advanced/README.md)**

---

### Phase 5: 포트폴리오 확장 (Month 7-9)

**목표**:
- ✅ Bill Evans 스타일 모델 추가
- ✅ Miles Davis 스타일 모델 추가
- ✅ 3개 모델 비교 데모 완성

**예산**: $150

**학습 자료**:
```
roadmap/phase5-portfolio/
├── README.md
├── bill_evans_model/           # Bill Evans 모델
│   ├── data/
│   ├── train.py
│   └── README.md
├── miles_davis_model/          # Miles Davis 모델
│   ├── data/
│   ├── train.py
│   └── README.md
├── comparison_demo/            # 3모델 비교 데모
│   ├── app.py
│   └── README.md
└── portfolio_guide.md          # 포트폴리오 작성 가이드
```

**체크리스트**:
- [ ] Bill Evans MIDI 수집 + 학습
- [ ] Miles Davis MIDI 수집 + 학습
- [ ] 3개 모델 비교 웹 데모
- [ ] GitHub README를 포트폴리오처럼 작성

**→ [Phase 5 시작하기](./phase5-portfolio/README.md)**

---

### Phase 6: AI 엔지니어 취업 준비 (Month 10-12)

**목표**:
- ✅ 포트폴리오 정리 및 문서화
- ✅ 이력서 작성 (프로젝트 중심)
- ✅ AI 엔지니어 포지션 지원
- ✅ 면접 준비

**예산**: $0

**학습 자료**:
```
roadmap/phase6-job/
├── README.md
├── resume_template.md          # 이력서 템플릿
├── cover_letter.md             # 자기소개서 가이드
├── portfolio_presentation.md   # 포트폴리오 설명법
├── interview_prep.md           # 면접 준비
└── company_list.md             # 지원할 회사 리스트
```

**체크리스트**:
- [ ] GitHub 프로필 정리 (README 작성)
- [ ] 이력서 작성 (프로젝트 중심)
- [ ] AI 스타트업 20곳 리스트업
- [ ] 주 3-5곳씩 지원
- [ ] 면접 준비 (기술 질문 + 프로젝트 설명)

**→ [Phase 6 시작하기](./phase6-job/README.md)**

---

## 💰 상세 예산 계획

### Month-by-Month 예산

| Month | 항목 | 비용 | 누적 |
|-------|------|------|------|
| 1 | 무료 (Colab 무료) | $0 | $0 |
| 2 | Runpod RTX 3090 (30h) | $30 | $30 |
| 3 | Colab Pro ($10) + Runpod ($40) | $50 | $80 |
| 4 | Runpod 실험 | $50 | $130 |
| 5 | Runpod 실험 | $50 | $180 |
| 6 | Runpod 실험 | $50 | $230 |
| 7 | Bill Evans 학습 | $50 | $280 |
| 8 | Miles Davis 학습 | $50 | $330 |
| 9 | 최종 실험 | $50 | $380 |
| 10-12 | 취업 준비 (무료) | $0 | $380 |

**총 투자**: $380 (약 50만원)
**ROI**: AI 엔지니어 초봉 4000-5000만원 → 약 100배 수익률

---

## 🎯 성공 지표 (KPI)

### Month 3 (최소 달성 목표)
- ✅ GitHub: 1개 프로젝트, 50+ commits
- ✅ HuggingFace: 1개 모델 업로드
- ✅ 데모: Gradio 웹 데모 작동

### Month 6 (중간 목표)
- ✅ GitHub: 3개 프로젝트, 200+ commits
- ✅ HuggingFace: 2개 모델, 100+ downloads
- ✅ 웹 서비스: Spring Boot + React + ML API

### Month 12 (최종 목표)
- ✅ GitHub: 5개 프로젝트, 500+ commits, 100+ stars
- ✅ HuggingFace: 3개 모델, 1000+ downloads
- ✅ 블로그: 10편 이상 기술 글
- ✅ 취업: AI 엔지니어 또는 백엔드 개발자 (AI 경험 인정)

---

## 🚀 시작하기

### 오늘 바로 할 수 있는 것

```bash
# 1. 이 저장소 clone
git clone <your-repo-url>
cd roadmap

# 2. Phase 1 시작
cd phase1-setup
cat README.md  # 읽어보기

# 3. Google Colab 접속
# https://colab.research.google.com/

# 4. 첫 번째 노트북 열기
# phase1-setup/01_colab_setup.md 따라하기
```

### 주간 학습 플랜 (예시)

**주중 (월-금)**: 백엔드 공부 (Java Spring)
- 저녁 2시간씩 Spring Boot 공부
- 목표: 3-6개월 후 백엔드 취업

**주말 (토-일)**: AI 프로젝트
- 토요일: 6시간 (학습 + 실험)
- 일요일: 4시간 (코드 정리 + 문서화)
- **주당 10시간** = 월 40시간 AI

**이 페이스면 충분히 가능해!**

---

## 📚 추가 리소스

### 무료 학습 자료

1. **PyTorch 기초**
   - PyTorch 공식 튜토리얼: https://pytorch.org/tutorials/
   - 한국어 튜토리얼: https://tutorials.pytorch.kr/

2. **Music AI**
   - Google Magenta: https://magenta.tensorflow.org/
   - Music Transformer 논문: https://arxiv.org/abs/1809.04281

3. **MIDI 처리**
   - pretty_midi 문서: https://craffel.github.io/pretty-midi/
   - mido 문서: https://mido.readthedocs.io/

### 커뮤니티

- Reddit: r/MachineLearning, r/WeAreTheMusicMakers
- Discord: AI Music Discord
- 한국: PyTorch 한국 사용자 모임
- 카카오톡: AI 오픈채팅방

---

## ❓ FAQ

**Q1: GPU 없어도 정말 가능해?**
- ✅ 네! Google Colab 무료 + 필요할 때만 Runpod 사용
- Month 1은 완전 무료, Month 2부터 월 $30-50

**Q2: 코딩 실력이 부족한데?**
- ✅ 복사-붙여넣기 가능한 코드 제공
- 주석으로 자세히 설명
- 막히면 커뮤니티에 질문

**Q3: 고졸이어도 취업 가능해?**
- ✅ 포트폴리오가 곧 학력
- HuggingFace 모델 3개 > 대학 학위
- 실제 작동하는 프로젝트가 증명

**Q4: 실패하면?**
- ✅ 실패 없음. 백엔드 취업은 별개로 진행
- AI 안 되면 백엔드로 가면 됨
- 하지만 이 로드맵 따라하면 90% 성공

**Q5: 혼자 할 수 있을까?**
- ✅ 이 로드맵이 그걸 위한 거야
- 단계별로 명확함
- 막히면 이슈 올리면 됨

---

## 🎵 마지막 메시지

**1년 후 너**:
- GitHub: 500+ commits, 100+ stars
- HuggingFace: 3개 모델, 1000+ downloads
- 포트폴리오: "찰리 파커 AI 만든 사람"
- 취업: AI 엔지니어 또는 AI 역량 있는 백엔드 개발자

**이게 너한테 가능한 미래야.**

**지금 시작하면 1년 후 여기 도착해.**

---

## 📞 문의 및 기여

- 이슈: GitHub Issues에 질문 남기기
- 기여: PR 환영!
- 블로그: 학습 과정을 블로그에 기록하면 더 좋음

---

**시작하자! → [Phase 1으로 이동](./phase1-setup/README.md)** 🚀
