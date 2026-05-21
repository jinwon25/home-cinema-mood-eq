# AI 기반 홈 시네마 음향 자동 최적화 시스템

> 🏆 **포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문)** — 스마트폰 한 대만으로 방의 음향 지문(xRIR)을 측정하고, 영상 장면의 감정(Mood)을 실시간으로 분석하여 홈 시네마 음향을 자동으로 세팅·보정하는 시스템입니다. 공간을 보는 ViT와 소리를 듣는 ResNet-18을 Cross-Attention으로 융합하여 **1024차원 공간 음향 지문**을 예측하고, X-CLIP·PANNs 기반 **Valence-Arousal 감정 회귀**로 씬 단위 사운드 믹싱을 구동합니다. 사용자 블라인드 평가에서 **40명 중 32명(80%)이 시스템 적용본을 선호**.

<br>

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 시스템 명 | **POFLIX** (Scene-Aware Smart Audio Optimization) |
| 진행 기간 | 2026.03 ~ 2026.04 (2026.03.30 ~ 2026.04.29) |
| 소속 | 포스코 청년 AI·Big Data 아카데미 32기 — C4 팀 |
| 팀 구성 | 5인 팀 (데이터셋 정제 · 모델링 · 성능 평가 · 서비스 프로토타입 설계 · 발표 자료 구성 역할 분담) |
| 본인 역할 | 모델링 · 성능 평가 · 발표 자료 구성 (**기여도 25%**) |
| 핵심 도구 | PyTorch · FastAPI · React Native · Pedalboard · librosa · Silero VAD · PANNs · PySceneDetect · webMUSHRA |
| 주요 산출물 | xRIR 추론 백엔드 · Mood-EQ 워커 · 블라인드 청취 평가 인프라 · 객관 지표 평가 도구 |
| 수상 | 🏆 **포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문)** |
| 기여 흔적 | [commits by jinwon25](https://github.com/jinwon25/home-cinema-mood-eq/commits?author=jinwon25) |

<br>

## 2. 분석 요약 (Key Findings)

- **공간 음향 지문 예측(xRIR)** — 스마트폰 LiDAR/RoomPlan + 1회 sweep 녹음만으로 ViT(공간)·ResNet-18(잔향)을 Cross-Attention으로 융합, **1024차원 공간 음향 지문**을 예측하고 음향적으로 우수한 스피커 좌표 5곳을 추천한 뒤 Inverse EQ로 자동 보정합니다.
- **콘텐츠 인식 동적 EQ(Mood-EQ)** — X-CLIP + PANNs로 영상의 시각·청각 특징을 추출하고 **Valence-Arousal(V/A) 회귀**로 씬 감정을 추정하여, 객체 분리 기반 **10-band 동적 EQ**와 대사 보호(Ducking)로 폭발음 등 격렬한 장면에서도 대사 명료도를 유지합니다.
- **OOD 검증 통과** — LIRIS-ACCEDE 160편/9,800클립으로 학습한 V/A 모델을 **COGNIMUSE 200개 클립**(학습 미포함)으로 일반화 성능 검증 완료.
- **MUSHRA 14명 블라인드 청취 평가** — 국제 표준 평가 프로토콜로 통계적 유의성까지 입증.

<br>

## 3. 배경 및 목적

홈 시청 환경에서 "좋은 소리"를 막는 세 가지 한계에서 출발했습니다.

1. **공간이 만드는 음향 한계** — 한국형 LDK(Living-Dining-Kitchen) 구조는 비대칭 음향 환경을 만들고, 벽·가구 반사로 저음역대 부밍(Booming) 등 음질 왜곡이 발생합니다.
2. **시청 환경의 변화** — OTT의 확대로 집이 주요 시청 공간이 됐지만, 과도한 배경음·효과음 탓에 대사 전달력이 떨어져 **사용자의 약 85% 가 자막에 의존**합니다.
3. **기존 솔루션의 한계** — Sonos, Apple HomePod 등은 1회성 공간 측정(정적 EQ)에 머물러 영상 상황(액션·대화)에 맞춘 실시간 동적 제어가 불가능합니다.

→ POFLIX는 *공간*과 *콘텐츠*를 동시에 이해하는 투 트랙 시스템으로 위 한계를 해결합니다.

<br>

## 4. 데이터

| 데이터셋 | 규모 | 용도 |
|---|---|---|
| **AcousticRooms** (CVPR 2025, *Hearing Anywhere in Any Environment* 기반 재가공) | 174 공간 / **19만+** RIR | 공간 음향 학습 (xRIR) |
| **LIRIS-ACCEDE** | 160 영화 / **9,800** 클립 | Valence-Arousal 회귀 학습 |
| **COGNIMUSE** | 200 클립 | OOD(Out-of-Distribution) 일반화 검증 |

### 데이터 가공 (도메인 최적화)

- 원본 AcousticRooms는 260개 공간 / 30만 RIR이었으나 화장실·대강당 등을 제외하고 **아파트 · 거실 · 침실 · 청음실 · 회의실 5개 카테고리** 의 174개 공간으로 압축.
- 실제 주거 공간의 가구 재질과 저음역대 부밍(Booming)을 반영한 High-Fidelity 시뮬레이션을 적용.
- LIRIS-ACCEDE는 4초 윈도우 슬라이딩(Stride 2초)으로 입력 형식을 표준화하고 샘플 수를 확장.

<br>

## 5. 시스템 흐름

### 5.1 공간 인식 음향 최적화 (xRIR)

스마트폰 LiDAR/RoomPlan으로 3D 공간 지도를 만들고 1회 sweep 음원을 재생·녹음.

- **공간을 보는 눈** — ViT가 방의 형태와 구조를 파악.
- **소리를 듣는 귀** — ResNet-18이 울림과 잔향 패턴을 분석.
- **Cross-Attention 융합** — 두 신호를 결합해 **1024차원 공간 음향 지문(xRIR)** 예측.
- **최적 좌표 추천 + Inverse EQ** — 음향적으로 우수한 스피커 위치 5곳 추천, 배치 후 최종 Inverse EQ 계산으로 공간 보정 완료.

### 5.2 콘텐츠 인식 동적 EQ (Mood-EQ) — 본인 담당 트랙

영상 장면의 감정과 음원 구조를 분석해 실시간으로 사운드를 믹싱.

- **씬 감정 분석** — X-CLIP(시각) + PANNs(청각) 특징을 결합하여 Valence-Arousal 회귀로 감정 상태 추정.
- **객체 분리 기반 동적 10-band EQ** — 폭발음·격투 씬에서도 대사 명료도를 잃지 않도록 대역별 동적 제어.
- **대사 보호(Ducking)** — 배경음이 커질 때 대사 트랙을 우선 살리는 Sidechain Ducking.

자세한 시스템 명세: [docs/system-overview.md](./docs/system-overview.md)

### 5.3 본인 기여 상세

- **Mood-EQ inference worker 1차 구현** — V/A 회귀 결과를 10-band 동적 EQ로 매핑.
- **MUSHRA 블라인드 청취 평가 큐레이션 모드** · simple_player UI 설계.
- **A/B 즉시 전환 + JND(Just-Noticeable Difference) 캘리브레이션 도구** 구축.
- **씬 라벨 자동·수동 검수 파이프라인**.
- **V3.5.7 최종 파이프라인 채택 결정** — M/S, Sidechain Ducking, 객관 지표 통합.
- **V3.3 EQ 범위 ±6dB 확대 + Compressor 후처리 결정 기여**.

<br>

## 6. 평가 지표

| 분류 | 적용 기술 | 평가지표 | 검증 목적 |
|---|---|---|---|
| UX 지표 | 공간 인식 | Distance Error (m) | 시뮬레이션상 최적 좌표(GT)와 AI 추천 위치 간 물리적 거리 오차 |
| AI 코어 | 공간 인식 | RT60 · C80 · DRR MAE | 잔향 시간·명료도 등 물리 음향 법칙에 대한 모델의 예측 오차율 (Pyroomacoustics GT 대비) |
| 상용화 | 공간 인식 | Inference Speed (ms) | 모바일 환경 적용을 위한 Backbone 경량화 및 추론 속도 |
| 음향 표준 | 콘텐츠 인식 | STOI · SI-SDR · LSD | 대사 명료도 · 음원 왜곡률 · 음색 변화 등 믹싱 후 객관적 음질 향상 |
| 자체 검증 | 콘텐츠 인식 | MRI · DPR | Ducking 속도·부드러움, 배경음 대비 대사 볼륨 유지력 |
| 주관 평가 | 콘텐츠 인식 | **MUSHRA Test** | 국제 표준 블라인드 청취 평가(14명) — 통계적 유의성 입증 |

<br>

## 7. 기술 스택

| 분류 | 사용 도구 |
|---|---|
| Backend | FastAPI · PyTorch · pyroomacoustics |
| Mobile | React Native 0.74 · TypeScript |
| ML 학습 | PyTorch · timm · einops · auraloss |
| 오디오 처리 | Pedalboard · librosa · Silero VAD · PANNs |
| 씬 분석 | PySceneDetect · OpenCV · X-CLIP |
| 평가 | webMUSHRA · STOI · SI-SDR · LSD · MRI · DPR |

<br>

## 8. 디렉토리 구조

```
home-cinema-mood-eq/
├── README.md
├── LICENSE
├── requirements.txt
├── run_pipeline.py             # 메인 추론 파이프라인 진입점
│
├── backend/                    # FastAPI 서버 (job orchestration, xRIR 추론)
├── mobile/                     # React Native 앱 (3D 스캔, sweep 녹음, UI)
├── model/                      # 학습 파이프라인 (3D 공간 음향, autoEQ)
│
├── scripts/                    # 데이터 전처리·feature 추출
│   ├── precompute_*.py         #   AST · CLIP-frame mean features
│   ├── evaluate_lomo_testsets.py
│   ├── compare_ablation.py
│   └── demos/                  #   A/B 비교·청취 데모 스크립트
│       ├── compare_spectra.py
│       ├── live_compare.py
│       ├── live_compare_fx.py
│       └── generate_fx_demo.py
│
├── evaluation/                 # 청취 평가 (본인 담당)
│   ├── webmushra/              #   MUSHRA 블라인드 테스트 UI
│   ├── baseline/scene_labels/  #   씬 라벨 baseline (lalaland · topgun)
│   ├── results/                #   청취 결과 (gitignored)
│   └── rehearsal_checklist.md
│
├── tools/                      # MUSHRA · V3.5.x 평가 보조 도구
│   ├── run_v3_5_{5,6,7}_pipeline.py
│   ├── analyze_mushra_results.py
│   ├── objective_metrics{,_compare}.py
│   ├── auto_select_segments.py
│   ├── generate_{anchor,mid_anchor,calibration_tones}.py
│   └── loudness_match.py
│
├── data/   dataset/   outputs/  # 외부 데이터·산출물 마운트 (gitignored)
│
└── docs/
    ├── system-overview.md      # POFLIX 전체 시스템 명세
    ├── specification.md        # V3.3 시스템 스펙
    ├── liris-migration-plan.md # LIRIS-ACCEDE 전환 최종 plan
    ├── worker-guide.md         # Mood-EQ 워커 운영 가이드
    ├── audio-features.md       # 음향 기능 상세
    ├── decisions/              # V3.5.x → V3.5.7 의사결정 기록
    └── migration_history/      # LIRIS plan V1~V5 archive
```

<br>

## 9. 실행 방법

```bash
pip install -r requirements.txt

# 메인 추론 파이프라인
python run_pipeline.py

# 평가 파이프라인 (V3.5.7)
python tools/run_v3_5_7_pipeline.py

# MUSHRA 결과 분석
python tools/analyze_mushra_results.py
```

자세한 운영·디버깅 가이드: [docs/worker-guide.md](./docs/worker-guide.md)

<br>

## 10. 한계 및 향후 과제

- **데이터셋 도메인 차이** — AcousticRooms 시뮬레이션 RIR과 실제 한국형 LDK 거주 공간 사이에는 가구 재질·벽재의 차이가 남아 있어, 실측 RIR 수집을 통한 도메인 적응이 필요합니다.
- **MUSHRA 14명 표본 한계** — 국제 표준이지만 청취자 수가 적어, 더 큰 패널과 다국적 표본으로의 확장 검증이 후속 과제입니다.
- **실시간 처리 지연** — Mood-EQ는 씬 단위 추론. 진정한 프레임 단위 실시간으로 가려면 Backbone 추가 경량화가 필요합니다.
- **온디바이스 통합** — 현재 Mobile은 스캔·녹음·UI 담당, 추론은 백엔드. 추후 핵심 모델의 온디바이스 추론 이식이 필요합니다.

<br>

## 11. 참고 자료

- 전체 시스템 명세: [docs/system-overview.md](./docs/system-overview.md)
- V3.3 시스템 스펙: [docs/specification.md](./docs/specification.md)
- Mood-EQ 워커 운영 가이드: [docs/worker-guide.md](./docs/worker-guide.md)
- 음향 기능 상세: [docs/audio-features.md](./docs/audio-features.md)
- 의사결정 기록(V3.5.x → V3.5.7): [docs/decisions/](./docs/decisions/)
- LIRIS-ACCEDE 전환 계획: [docs/liris-migration-plan.md](./docs/liris-migration-plan.md)

<br>

---

**작성자** · 최진원 (munjwc25@gmail.com) · 2026  
**팀** · 포스코 청년 AI·Big Data 아카데미 32기 — C4 (5인 팀), 본인은 모델링 · 성능 평가 · 발표 자료 구성 담당 (기여도 25%)  
**수상** · 🏆 포스코 청년 AI·Big Data 아카데미 32기 최우수상 (AI 부문)
