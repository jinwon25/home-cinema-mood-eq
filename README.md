# POFLIX — AI 기반 홈 시네마 음향 자동 최적화 시스템

> 🏆 **포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문) 수상작** — 스마트폰 한 대로 방의 음향 지문(xRIR)을 측정하고, 영상 장면의 감정을 실시간 분석해 홈 시네마 사운드를 자동 보정하는 시스템입니다. **ConvNeXT**(공간 시각) · **ResNet-18**(음향)을 **Cross-Attention**으로 융합한 **1024차원 공간 음향 지문**, X-CLIP + PANNs를 **Gate Network**로 결합한 **Valence-Arousal + Mood Head(K=7)** 감정 회귀, **Dual-Layer EQ + Silero VAD 기반 Dialogue Protection**까지 투 트랙(공간·콘텐츠)으로 작동합니다. 블라인드 A/B 청취 평가에서 **40명 중 32명(80%)이 시스템 적용본을 선호**했고, 공간 백본은 ViT → ConvNeXT 교체로 **C50 Error 유의 개선(p=0.0085)** 을 확인했으며, 감정 회귀 모델은 **CCC 0.3603** 으로 머신러닝 SOTA(Baruah 2017 CCC 0.200)를 큰 폭으로 상회하고 딥러닝 SOTA(Gan 2017 CCC 0.414)에 근접합니다.

<br>

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 시스템명 | **POFLIX** (Scene-Aware Smart Audio Optimization) |
| 진행 기간 | 2026.03.30 ~ 2026.04.29 |
| 소속 | 포스코 청년 AI·Big Data 아카데미 32기 — C4 (5인 팀) |
| 본인 역할 | 모델링 · 성능 평가 · 발표 자료 구성 (**기여도 25%**) |
| 핵심 도구 | PyTorch · FastAPI · React Native · Pedalboard · librosa · Silero VAD · X-CLIP · PANNs · ConvNeXT · ResNet-18 · PySceneDetect |
| 주요 산출물 | xRIR 추론 백엔드 · Mood-EQ 워커 · 블라인드 A/B 청취 평가 인프라 · 객관 지표 평가 도구 |
| 수상 | 🏆 **포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문)** |
| 기여 흔적 | [commits by jinwon25](https://github.com/jinwon25/home-cinema-mood-eq/commits?author=jinwon25) |

<br>

## 2. 분석 요약 (Key Findings)

- **공간 음향 지문 예측(xRIR)** — 스마트폰 LiDAR/RoomPlan + 1회 sweep 녹음으로, **ConvNeXT**(공간 시각) · **ResNet-18**(음향)을 **Cross-Attention**으로 융합해 **1024차원 공간 음향 지문**을 예측. 음향적으로 우수한 스피커 좌표 5곳을 추천한 뒤 Inverse EQ로 자동 보정. → ViT baseline 대비 **C50 Error**에서 유의한 개선(p=0.0085) 확인 후 ConvNeXT 최종 채택.
- **콘텐츠 인식 동적 EQ(Mood-EQ)** — X-CLIP(시각, 5120d)·PANNs(청각, 2048d) 특징을 **Gate Network**로 가중 융합(Concat·GMU 대비 채택, ablation 근거)하여 **Valence-Arousal 회귀 + Mood Head(K=7)**로 씬 감정 추정. → **Dual-Layer EQ**(Layer1 10-band Peaking EQ + Layer2 Shelf·Reverb·Limiter)로 변환, **Silero VAD 기반 Dialogue Protection**으로 폭발음 씬에서도 대사 명료도 유지.
- **OOD 일반화 검증** — LIRIS-ACCEDE(160편/9,800 클립) 학습 모델을 **COGNIMUSE 200 클립**(학습 미포함)에서 평가 → mean CCC **0.3781** (V 0.3113 · A 0.4449), in-distribution 대비 손실 없음.
- **블라인드 A/B 청취 평가** — 3개 반 **총 40명** 대상으로 원본 vs 시스템 적용본 청취 후 선호 응답: A반 14명 중 10명(71%) · B반 15명 중 14명(93%) · C반 11명 중 8명(73%) → **합계 32/40 = 80% 선호**.

<br>

## 3. 배경 및 목적

홈 시청 환경에서 "좋은 소리"를 막는 세 가지 한계에서 출발했습니다.

1. **공간이 만드는 음향 한계** — 한국형 LDK(Living-Dining-Kitchen) 구조는 비대칭 음향 환경을 만들고, 벽·가구 반사로 저음역대 부밍(Booming)이 발생합니다.
2. **시청 환경의 변화** — OTT 확대로 집이 주요 시청 공간이 됐지만, 과도한 배경음·효과음 탓에 대사 전달력이 떨어져 **사용자의 약 85%가 자막에 의존**합니다.
3. **기존 솔루션의 한계** — Sonos, Apple HomePod 등은 1회성 공간 측정(정적 EQ)에 머물러 영상 상황(액션·대화)에 맞춘 실시간 동적 제어가 불가능합니다.

→ POFLIX는 *공간*과 *콘텐츠*를 동시에 이해하는 투 트랙 시스템으로 위 한계를 해결합니다.

<br>

## 4. 데이터

| 데이터셋 | 원본 → 정제 | 용도 |
|---|---|---|
| **AcousticRooms** (CVPR 2025, *Hearing Anywhere in Any Environment* 기반 재가공) | 260 공간 / 30만 RIR → **174 공간 / 19만+ RIR** (홈 시네마 5개 카테고리: 아파트·거실·침실·청음실·회의실) | 공간 음향 학습 (xRIR) |
| **LIRIS-ACCEDE** | 160 영화 / **9,800 클립** (Train 64 / Val 16 / Test 80, 4초 윈도우 · stride 2초) | Valence-Arousal 회귀 학습 |
| **COGNIMUSE** | 200 클립 | OOD(Out-of-Distribution) 일반화 검증 |

→ 실제 주거 공간의 가구 재질과 저음역대 부밍을 반영한 High-Fidelity 시뮬레이션 적용.

> ⚠️ AcousticRooms · LIRIS-ACCEDE · COGNIMUSE는 모두 외부 라이선스 데이터셋이므로 본 저장소에는 포함하지 않습니다. 각각 원본 논문 저자 배포처 / 공식 배포처에서 라이선스 동의 후 다운로드해야 합니다.

<br>

## 5. 시스템 흐름

### 5.1 공간 인식 음향 최적화 (xRIR 트랙)

스마트폰 LiDAR/RoomPlan으로 3D 공간 지도를 만들고 1회 sweep 음원을 재생·녹음.

- **공간을 보는 눈 (ConvNeXT)** — 방의 형태·구조 파악. ViT를 baseline으로 비교 평가 후 C50 Error 유의 개선(p=0.0085)을 근거로 ConvNeXT 채택.
- **소리를 듣는 귀 (ResNet-18)** — 잔향·반사 패턴 분석.
- **Cross-Attention 융합** — 두 모달리티를 결합해 **1024차원 공간 음향 지문(xRIR)** 예측.
- **최적 좌표 추천 + Inverse EQ** — 음향적으로 우수한 스피커 위치 5곳을 추천, 배치 후 Inverse EQ 계산으로 공간 보정 완료.

### 5.2 콘텐츠 인식 동적 EQ (Mood-EQ 트랙) — 본인 담당

영상 장면의 감정과 음원 구조를 분석해 사운드를 씬 단위로 믹싱.

- **씬 감정 분석** — X-CLIP(시각, 5120d, 시간 맥락) + PANNs(청각, 2048d + Audio Projection 512d)
- **Gate Network 융합** — 두 모달리티의 중요도를 씬마다 게이트로 가중 (총 10240d). Concat / GMU와의 ablation 비교 후 채택.
- **VA + Mood Head(K=7)** — 연속 Valence-Arousal 회귀와 7-클래스 mood 분류를 동시에 출력해 씬 타임라인에 집계.
- **Dual-Layer EQ** —
  - **Layer 1: 10-band Peaking EQ** — 감정→EQ 매핑(ISO R-40 octave band 기반)으로 학술 contribution.
  - **Layer 2: Shelf · Reverb · Limiter** — 지각 한계(JND ~1 dB) 위로 증폭하는 perceptual amplifier.
- **VAD 기반 Dialogue Protection** — Silero VAD가 측정한 씬 내 대사 비율 `density`로 voice-critical 대역(1·2·4 kHz) gain을 선형 감쇠 (`g_eff = g_orig · (1 − (1 − α_d)·density)`). Layer 2의 reverb tail도 대사 구간만 30 ms raised-cosine crossfade로 bypass.

자세한 시스템 명세 — [docs/system-overview.md](./docs/system-overview.md), [docs/audio-features.md](./docs/audio-features.md)

### 5.3 본인 기여 상세

- **Mood-EQ inference worker 구현** — V/A 회귀 결과를 10-band 동적 EQ로 매핑.
- **블라인드 A/B 청취 평가 인프라** — webMUSHRA 기반 큐레이션 모드, simple_player UI, A/B 즉시 전환 + JND(Just-Noticeable Difference) 캘리브레이션 도구 설계.
- **씬 라벨 자동·수동 검수 파이프라인** 구축.
- **V3.5.7 최종 파이프라인 채택 결정** — M/S 처리, VAD 기반 dialogue bypass, 객관 지표 통합.
- **V3.3 EQ 범위 ±6 dB 확대 + Limiter 후처리 결정** 기여.

<br>

## 6. 평가 지표 및 결과

### 6.1 공간 인식 (xRIR)

ISO 3382 기반 잔향 음향 지표 + UX/상용화 지표를 함께 측정.

| 분류 | 지표 | 검증 목적 |
|---|---|---|
| AI 코어 | **EDT** (Early Decay Time) | 초기 반사음 감쇠 시간 (PyRoomAcoustics 기준값 대비) |
| AI 코어 | **C50** (Clarity 50ms) | 직접음+초기 반사음 vs 후기 잔향 비율 (명료도) |
| AI 코어 | **RT60** (= T60, Reverberation Time) | 60 dB 감쇠 시간 (전체 잔향) |
| UX 지표 | **공간 일치도 (정규화 거리)** | grid_best 거리를 방 부피의 세제곱근으로 정규화 — 시뮬레이션상 최적 좌표(GT)와 AI 추천 위치 간 부합도 (낮을수록 우수) |
| 상용화 | **추론 시간 (초)** | PyRoomAcoustics 시뮬레이터 대비 추론 속도 |

**단일 RIR(k=1) baseline 성능 — 원본 논문(k=8) 대비**

| 지표 | 원본 (k=8) | 우리 (k=1) | 변화 | 해석 |
|---|---:|---:|:-:|---|
| EDT | 0.0552 | **0.0381** | 개선 | 초기 반사음 추정이 더 정확 |
| C50 | 1.3605 | **1.1227** | 개선 | 명료도 예측 오차 감소 |
| T60 | 9.6940 | 13.0114 | 악화 | ref RIR이 1개뿐이라 전체 잔향 추정은 한계 (한계 §10 참고) |

**ViT vs ConvNeXT 백본 비교**

| 지표 | ConvNeXT Error | p-value | 결정 |
|---|---:|---:|---|
| EDT | 0.0373 | 0.0839 | 무의미 |
| **C50** | **1.0827** | **0.0085** | **유의 — ConvNeXT 채택 근거** |
| T60 | 14.0089 | 0.0001 | 유의 |

→ C50에서 통계적으로 유의한 개선을 보여 ConvNeXT를 최종 백본으로 선정.

**실측 룸 평가 — 공간 일치도 + 추론 속도**

| 룸 | 공간 일치도 (정규화 거리) | 우리 추론 시간 | PyRoomAcoustics | 속도 차 |
|---|---:|---:|---:|---:|
| C반 회의실 | **0.11** | 4.0 s | 48.0 s | **12×** |
| 4층 휴게실 | 0.32 | 5.8 s | 184.8 s | **32×** |
| 국제관 | 0.37 | 3.3 s | 59.5 s | **18×** |

→ PyRoomAcoustics 대비 약 **12~32배 단축**되어 모바일 실시간 추론에 적합. 공간 일치도는 회의실(0.11)에서 가장 우수.

**음향 근접성 검증** — 측정 RIR을 PyRoomAcoustics로 시뮬레이션한 기준값(PRA) 대비

| 지표 | PRA 기준 | Ours | 차이 |
|---|---:|---:|---:|
| EDT | 0.71 | 0.75 | +0.04 |
| RT60 (= T60) | 1.86 | 1.80 | −0.06 |
| C50 | 5.02 | 4.86 | −0.16 |

→ 실측 룸에서도 PRA reference와 근접한 음향 파라미터를 예측 — 시뮬레이션 도메인을 벗어난 실측 데이터에서도 안정적으로 작동.

### 6.2 콘텐츠 인식 (Mood-EQ)

**헤드라인 지표 — CCC** (Lin 1989, Concordance Correlation Coefficient)

| 평가 | CCC mean | CCC_V | CCC_A | Pearson mean | MAE mean |
|---|---:|---:|---:|---:|---:|
| **LIRIS-ACCEDE (in-distribution)** | **0.3603** | 0.3895 | 0.3312 | 0.4034 | 0.3071 |
| **COGNIMUSE (OOD 일반화)** | **0.3781** | 0.3113 | 0.4449 | — | — |

→ in-distribution에서 OOD로 성능이 떨어지지 않고 오히려 유지 — 모델이 LIRIS-ACCEDE 분포에 과적합되지 않았다는 신호.

**기존 연구 비교**

| 모델 | CCC | 비고 |
|---|---:|---|
| Baruah 2017 (ML SOTA) | 0.200 | 머신러닝 기반 |
| **POFLIX (우리)** | **0.360** | Pearson 0.403 |
| Gan 2017 (DL SOTA) | 0.414 | 딥러닝 기반 |

→ ML SOTA 대비 큰 격차로 우월, DL SOTA에 근접한 성능. 학회 데모/제품 수준에서 경쟁력 있는 결과.

**모델 선택 ablation (CCC 기준)**

| 비교 | CCC 차이 | p-value | 결정 근거 |
|---|---:|---:|---|
| X-CLIP vs CLIPMean | +0.011 | 0.470 | 무의미 — 시간 맥락 활용 위해 X-CLIP 채택 |
| **PANNs vs AST** | **+0.052** | **0.0224** | **유의 — PANNs 채택** |
| **Gate vs Concat / GMU** | **+0.006** | **0.590** | Concat·GMU와 무차별 — 학습 안정성 우위로 Gate 채택 |
| VA + Mood vs VA only | +0.010 | 0.220 | 무의미하나 mood 출력이 EQ preset 선택에 직접 사용 → 채택 |

### 6.3 오디오 품질 보조지표 (post-processing)

V/A·CCC는 모델 회귀 정확도 지표이고, EQ 처리 결과의 음향적 영향을 별도로 측정하기 위해 다음 객관 지표를 [`tools/objective_metrics.py`](./tools/objective_metrics.py)에서 산출.

| 지표 | 의미 |
|---|---|
| STOI / ESTOI | Short-Time Objective Intelligibility (Taal 2010, Jensen 2016) — 대사 명료도 |
| SI-SDR | Scale-Invariant SDR (Le Roux 2019) — 신호 왜곡도 |
| LSD | Log-Spectral Distance (dB) — 음색 변화 |
| **DPR** (custom) | Dialogue Protection Ratio — 1–4 kHz 대역의 대사 활성 구간 RMS 변화 |
| **MRI** (custom) | Music Restoration Index — 대사 비활성 구간 5–15 kHz 변화 |
| IACC / SEI | Spatial expansion 측정 (M/S 처리 검증) |

→ 이 지표들은 처리 전후 오디오의 **객관적 음향 변화**를 보고하는 보조 도구이며, 모델의 핵심 성능 지표는 위 §6.2의 **CCC**.

### 6.4 사용자 평가 — 블라인드 A/B 청취 테스트

동일 영상의 **원본 vs 시스템 적용본**을 블라인드로 청취 후 선호 선택. 평가 기준: 대사 명료도 · 장면 몰입감 · 전반적 사운드 만족도.

| 그룹 | 인원 | 원본 선호 | 시스템 선호 | 선호율 |
|---|:-:|:-:|:-:|---:|
| A반 | 14 | 4 | 10 | 71% |
| B반 | 15 | 1 | 14 | 93% |
| C반 | 11 | 3 | 8 | 73% |
| **합계** | **40** | **8** | **32** | **80%** |

→ 3개 반 모두에서 시스템 적용본이 다수 선호. 표본 분산을 고려해도 일관된 우세.

<br>

## 7. 활용 시나리오

POFLIX의 두 트랙(공간 인식 + 콘텐츠 인식)을 상용 환경에 매핑한 활용 사례.

| 시나리오 | 활용 트랙 | 제품 매핑 |
|---|---|---|
| **AVR / 사운드바 자동 캘리브레이션** | xRIR — 1024d 공간 음향 지문 + Inverse EQ | 별도 측정 마이크 없이 스마트폰 1회 sweep으로 룸 보정 |
| **OTT 플랫폼 실시간 음향 최적화** | Mood-EQ — 씬 감정 분석 + Dual-Layer EQ | Netflix · Disney+ 등에서 콘텐츠 감정에 동적으로 사운드 믹싱 |
| **청각 보조 — 대사 명료도 강화 모드** | Silero VAD 기반 Dialogue Protection (α_d 강도 조절) | 청각 약자 대상 영화·드라마 대사 명료도 자동 강화 |
| **B2B — 멀티플렉스 / 홈시어터 사전 캘리브레이션** | xRIR + 음향 근접성 검증 | 시뮬레이션 도메인 밖 실측 룸에서도 PRA 기준값에 근접 (§6.1) |

→ 블라인드 A/B 80% 선호율(§6.4)이 위 시나리오의 시장 수용성에 대한 정량 근거가 됨.

<br>

## 8. 기술 스택

| 분류 | 사용 도구 |
|---|---|
| Backend | FastAPI · PyTorch · pyroomacoustics |
| Mobile | React Native 0.74 · TypeScript (LiDAR/RoomPlan 스캔, sweep 녹음) |
| ML 학습 | PyTorch · timm · einops · auraloss |
| 모델 백본 | ConvNeXT (xRIR 공간) · ResNet-18 (xRIR 음향) · X-CLIP · PANNs CNN14 |
| 오디오 처리 | Pedalboard · librosa · **Silero VAD** |
| 씬 분석 | PySceneDetect · OpenCV · X-CLIP |
| 평가 | webMUSHRA (도구) · 블라인드 A/B 청취 평가 · STOI · SI-SDR · LSD · DPR · MRI |

<br>

## 9. 디렉토리 구조

```
home-cinema-mood-eq/
├── README.md
├── LICENSE
├── requirements.txt
├── run_pipeline.py             # 메인 추론 파이프라인 진입점
│
├── backend/                    # FastAPI 서버 (job orchestration, xRIR 추론)
├── mobile/                     # React Native 앱 (3D 스캔, sweep 녹음, UI)
├── model/                      # 학습 파이프라인 (xRIR · autoEQ)
│   └── autoEQ/train_pseudo/    #   ablation: model_base · model_concat · model_gmu · model_ast · model_ast_gmu · model_clip_framemean
│
├── scripts/                    # 데이터 전처리·feature 추출
│   ├── precompute_*.py         #   AST · CLIP-frame mean features
│   ├── evaluate_lomo_testsets.py
│   ├── compare_ablation.py
│   └── demos/                  #   A/B 비교·청취 데모
│       ├── compare_spectra.py · live_compare.py · live_compare_fx.py · generate_fx_demo.py
│
├── evaluation/                 # 청취 평가
│   ├── webmushra/              #   블라인드 A/B 테스트 UI
│   ├── baseline/scene_labels/  #   씬 라벨 baseline (lalaland · topgun)
│   ├── results/                #   청취 결과 (gitignored)
│   └── rehearsal_checklist.md
│
├── tools/                      # 평가 보조 도구
│   ├── run_v3_5_{5,6,7}_pipeline.py
│   ├── analyze_mushra_results.py
│   ├── objective_metrics{,_compare}.py   # STOI · SI-SDR · LSD · DPR · MRI · IACC · SEI · CAS
│   ├── auto_select_segments.py
│   ├── generate_{anchor,mid_anchor,calibration_tones}.py
│   └── loudness_match.py
│
├── data/   dataset/   outputs/  # ⚠️ gitignored (외부 데이터·산출물 마운트)
│
└── docs/
    ├── POFLIX_발표자료.pdf       # 최종 발표자료 (Source of Truth)
    ├── system-overview.md
    ├── specification.md          # V3.3 시스템 스펙
    ├── audio-features.md         # Dual-Layer EQ · VAD dialogue protection 상세
    ├── liris-migration-plan.md
    ├── worker-guide.md
    ├── decisions/                # V3.5.x → V3.5.7 의사결정 기록
    └── migration_history/
```

<br>

## 10. 실행 방법

```bash
pip install -r requirements.txt

# 메인 추론 파이프라인
python run_pipeline.py

# 평가 파이프라인 (V3.5.7)
python tools/run_v3_5_7_pipeline.py

# 객관 지표 측정 (STOI · SI-SDR · LSD · DPR · MRI ...)
python tools/objective_metrics.py
```

자세한 운영·디버깅 가이드: [docs/worker-guide.md](./docs/worker-guide.md)

<br>

## 11. 한계 및 향후 과제

- **데이터셋 도메인 차이** — AcousticRooms 시뮬레이션 RIR과 실제 한국형 LDK 거주 공간 사이에는 가구 재질·벽재 차이가 남아 있어, 실측 RIR 수집을 통한 도메인 적응이 필요.
- **단일 RIR(k=1) 제약 — T60 추정** — 본 시스템은 sweep 1회 측정을 전제로 하므로 ref RIR 수가 적어 전체 잔향 시간(T60)의 추정이 원본 k=8 대비 악화(9.69→13.01). EDT·C50(초기 반사음)은 개선되었으나, 후기 잔향까지 정확히 잡으려면 다지점 측정 또는 모델 개선이 필요.
- **블라인드 A/B 표본 한계** — 총 40명(3개 반)이라는 표본 규모는 통계적 검증에 한계가 있어, 더 큰 패널과 다국적 표본으로의 확장 검증이 후속 과제.
- **실시간 처리 지연** — Mood-EQ는 씬 단위 추론. 진정한 프레임 단위 실시간으로 가려면 Backbone 추가 경량화가 필요.
- **온디바이스 통합** — 현재 Mobile은 스캔·녹음·UI 담당, 추론은 백엔드. 추후 핵심 모델의 온디바이스 추론 이식이 필요.

<br>

## 12. 참고 자료

- 발표자료(PDF): [docs/POFLIX_발표자료.pdf](./docs/POFLIX_발표자료.pdf)
- 전체 시스템 명세: [docs/system-overview.md](./docs/system-overview.md)
- Dual-Layer EQ · VAD Dialogue Protection 상세: [docs/audio-features.md](./docs/audio-features.md)
- V3.3 시스템 스펙: [docs/specification.md](./docs/specification.md)
- Mood-EQ 워커 운영 가이드: [docs/worker-guide.md](./docs/worker-guide.md)
- 의사결정 기록(V3.5.x → V3.5.7): [docs/decisions/](./docs/decisions/)
- LIRIS-ACCEDE 전환 계획: [docs/liris-migration-plan.md](./docs/liris-migration-plan.md)

<br>

---

**작성자** · 최진원 (munjwc25@gmail.com) · 2026  
**팀** · 포스코 청년 AI·Big Data 아카데미 32기 — C4 (5인 팀), 본인은 모델링 · 성능 평가 · 발표 자료 구성 담당 (기여도 25%)  
**수상** · 🏆 포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문)
