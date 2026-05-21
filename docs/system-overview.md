# 🍿 POFLIX: AI 기반 홈 시네마 음향 자동 최적화 시스템
> **Scene-Aware Smart Audio Optimization** > 당신의 거실을 완벽한 영화관으로. 공간의 구조와 영상의 감정을 AI가 동시에 분석하여 최적의 홈 시네마 음향을 자동으로 세팅해 주는 시스템입니다.

## 📌 Project Background & Necessity
* **공간이 만드는 음향 한계:** 한국형 LDK(Living-Dining-Kitchen) 구조는 비대칭 음향 환경을 초래하며, 벽과 가구 반사로 인한 음질 왜곡(Booming)이 발생합니다.
* **콘텐츠 시청 환경의 변화:** OTT의 확대로 집이 주요 시청 공간이 되었으나, 과도한 배경음과 효과음으로 인해 대사 전달력이 저하되어 **사용자의 85%가 자막에 의존**하고 있습니다.
* **기존 솔루션의 한계:** Sonos, Apple HomePod 등 기존 기기들은 1회성 공간 측정(정적 EQ)에 그치며, 영상의 상황(액션, 대화 등)에 맞춘 실시간 동적 제어는 불가능합니다.

## ✨ Key Features & Architecture
본 프로젝트는 크게 **[기술 1] 공간 인식**과 **[기술 2] 콘텐츠 인식**의 투 트랙(Two-Track)으로 작동합니다.

### 1. 기술 1: 공간 인식 기반 음향 최적화 (Spatial Audio Optimization)
휴대폰 센서와 1회의 스윕(Sweep) 음원 재생만으로 방의 음향 지문을 파악하고 최적의 스피커 배치를 추천합니다.
* **공간 및 음향 데이터 수집:** 스마트폰 LiDAR/RoomPlan으로 3D 지도를 생성하고, 테스트 음원(Sweep)을 녹음.
* **멀티모달 음향 예측 (xRIR):** * 공간을 보는 눈 (ConvNeXT): 방의 형태와 구조 파악 — ViT를 baseline으로 비교 평가 후 C50 Error 유의 개선(p=0.0085)을 근거로 ConvNeXT 채택
  * 소리를 듣는 귀 (ResNet-18): 울림의 패턴과 잔향 분석
  * Cross-Attention 융합을 통해 1024차원의 공간 음향 지문(xRIR) 예측.
* **최적 배치 가이드 & 자동 보정:** 물리적으로 음향이 가장 훌륭한 5곳의 좌표를 추천하고, 배치 후 최종 Inverse EQ를 계산하여 공간 보정을 마무리합니다.

### 2. 기술 2: 콘텐츠 인식 기반 자동 EQ (Content-Aware Dynamic EQ)
영상 장면의 분위기와 대사를 분석하여 실시간으로 사운드를 믹싱합니다.
* **실시간 감정(Mood) 분석:** X-CLIP(시각, 5120d)과 PANNs(청각, 2048d)로 영상의 시각/청각적 특징을 추출하고, **Gate Network**로 두 모달리티의 중요도를 씬마다 가중 융합(Concat·GMU 대비 ablation 채택). **Valence-Arousal 회귀 + Mood Head(K=7)** 듀얼 출력으로 감정 상태를 추정합니다.
* **Dual-Layer EQ:**
  * **Layer 1 — 10-band Peaking EQ:** ISO R-40 octave band 기반 10개 주파수(31.5~16k Hz)에 감정→EQ 매핑 적용 (학술 contribution).
  * **Layer 2 — Shelf · Reverb · Limiter:** 지각 한계(JND ~1 dB) 위로 증폭하는 perceptual amplifier.
* **VAD 기반 Dialogue Protection:** Silero VAD가 측정한 씬 내 대사 비율(`density`)로 voice-critical 대역(1·2·4 kHz)의 EQ gain을 선형 감쇠(`g_eff = g_orig · (1 − (1 − α_d)·density)`). Reverb tail도 대사 구간만 30 ms raised-cosine crossfade로 bypass.

## 🗂 Dataset Processing
연구실 환경이 아닌 실제 상용 홈 시네마 환경에 맞추기 위해, 글로벌 논문의 데이터셋을 도메인에 맞게 전면 재가공했습니다.

### 1. 공간 음향 데이터셋 (AcousticRooms Refinement)
* 원본 논문(*Hearing Anywhere in Any Environment, CVPR 2025*)의 260개 공간, 30만 개 RIR 데이터를 필터링.
* **타겟 도메인 최적화:** 화장실, 대강당 등을 제외하고 **아파트, 거실, 침실, 청음실, 회의실 등 5개 카테고리의 174개 공간, 19만+ 고음질 RIR** 데이터로 압축.
* 실제 주거 공간의 가구 재질과 저음역대 부밍(Booming) 현상을 반영한 High-Fidelity 시뮬레이션 적용.

### 2. 감정 분석 데이터셋 (LIRIS-ACCEDE)
* 총 160편의 영화, 9,800개 클립을 기반으로 한 V/A(Valence-Arousal) 모델 학습.
* 4초 길이의 윈도우 슬라이딩(Stride 2초) 기법을 적용하여 입력 형식을 표준화하고 샘플 수를 확장.
* **OOD(Out-of-Distribution) 평가:** COGNIMUSE 데이터셋(200개 클립)을 활용하여, 학습에 쓰이지 않은 새로운 영화에 대한 일반화 성능 검증 완료.

## 📐 Evaluation Metrics
POFLIX 시스템의 신뢰성을 증명하기 위해 다음과 같은 평가 지표를 활용합니다.

| 분류 | 적용 기술 | 평가지표 (Metrics) | 검증 목적 |
| :--- | :--- | :--- | :--- |
| **UX 지표** | 기술 1 | **공간 일치도 (정규화 거리)** | 시뮬레이션상 최적 좌표(GT)와 AI 추천 위치 간의 부합도 — grid_best 거리를 방 부피의 세제곱근으로 정규화 (낮을수록 우수). 실측 3개 룸 결과: 0.11 / 0.32 / 0.37 |
| **AI 코어** | 기술 1 | **EDT, C50, RT60 (= T60)** | ISO 3382 기반 잔향 음향 지표 — 초기 반사음 감쇠 시간, 명료도, 60 dB 감쇠 시간을 PyRoomAcoustics 기준값 대비 비교 |
| **상용화** | 기술 1 | **추론 시간 (초)** | PyRoomAcoustics 시뮬레이터 대비 약 12~32배 단축 (실측 룸 3종: 4.0/5.8/3.3 s vs PRA 48.0/184.8/59.5 s) |
| **헤드라인**| 기술 2 | **CCC** (Concordance Correlation Coefficient, Lin 1989) | 예측 V/A 와 실제 V/A 의 일치도 — LIRIS-ACCEDE에서 mean 0.3603, COGNIMUSE(OOD) 0.3781 |
| **음향 보조**| 기술 2 | **STOI, SI-SDR, LSD, DPR, MRI** | EQ 처리 전후 오디오의 객관적 음향 변화 — 대사 명료도·왜곡·음색·대사 보호·음악 복원 |
| **주관 평가**| 기술 2 | **블라인드 A/B 청취 평가** | 원본 vs 시스템 적용본 블라인드 비교 청취 — 3개 반 총 40명에서 32명(80%) 시스템 선호 (A반 71% · B반 93% · C반 73%) |


---
*Developed by POSCO AIㆍBigData Academy 32nd C4 @ 2026*
