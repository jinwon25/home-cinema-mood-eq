# 홈 시네마 음향 자동 최적화 — POFLIX

방의 공간 음향과 영상의 감정을 AI가 동시에 분석해 홈 시네마 음향을 자동으로 세팅하는 시스템.
Scene-Aware Smart Audio Optimization.

## 진행 기간
2026.02 ~ 2026.04 (POSCO AI·BigData Academy 32기 C4 팀 프로젝트)

## 역할
4인 팀 · 콘텐츠 인식 동적 EQ(Mood-EQ) 워커와 평가 인프라 담당

- Mood-EQ inference worker 1차 구현 — V/A 회귀 결과를 10-band 동적 EQ로 매핑
- MUSHRA 블라인드 청취 평가 큐레이션 모드·simple_player UI 설계
- A/B 즉시 전환 + JND(Just-Noticeable Difference) 캘리브레이션 도구
- 씬 라벨 자동·수동 검수 파이프라인
- V3.5.7 최종 파이프라인 채택 결정 (M/S, Sidechain Ducking, 객관 지표 통합)
- V3.3 EQ 범위 ±6dB 확대 + Compressor 후처리 결정 기여

기여 흔적: [commits by jinwon25](https://github.com/jinwon25/home-cinema-mood-eq/commits?author=jinwon25)

## 데이터
| 데이터셋 | 규모 | 용도 |
|---|---|---|
| AcousticRooms (CVPR 2025) | 174 공간 / 19만+ RIR | 공간 음향 학습 (xRIR) |
| LIRIS-ACCEDE | 160 영화 / 9,800 클립 | Valence-Arousal 회귀 학습 |
| COGNIMUSE | 200 클립 | OOD 일반화 검증 |

## 시스템 흐름

### 1. 공간 인식 음향 최적화 (xRIR)
스마트폰 LiDAR/RoomPlan과 1회 sweep 녹음만으로 방의 음향 지문을 만들고 최적 스피커 좌표를 추천.
- ViT(공간) × ResNet-18(잔향) → Cross-Attention 융합 → 1024차원 공간 음향 지문
- 최적 스피커 좌표 5곳 추천 + 배치 후 Inverse EQ 자동 보정

### 2. 콘텐츠 인식 동적 EQ (Mood-EQ) — 담당 트랙
영상 장면의 감정과 음원 구조를 분석해 실시간으로 사운드를 믹싱.
- X-CLIP + PANNs → Valence-Arousal 회귀로 씬 감정 분석
- 객체 분리 기반 10-band 동적 EQ
- 대사 보호 Ducking — 폭발음·격투 씬에서도 대사 명료도 유지

자세한 시스템 설명: [docs/system-overview.md](./docs/system-overview.md)

## 기술 스택
| 영역 | 도구 |
|---|---|
| Backend | FastAPI, PyTorch, pyroomacoustics |
| Mobile | React Native 0.74, TypeScript |
| ML 학습 | PyTorch, timm, einops, auraloss |
| 오디오 처리 | Pedalboard, librosa, Silero VAD, PANNs |
| 씬 분석 | PySceneDetect, OpenCV, X-CLIP |
| 평가 | webMUSHRA, STOI · SI-SDR · LSD · MRI · DPR |

## 평가 지표 (담당 트랙)
| 분류 | 지표 | 검증 목적 |
|---|---|---|
| 음향 표준 | STOI, SI-SDR, LSD | 대사 명료도, 음원 왜곡, 음색 변화 |
| 자체 검증 | MRI, DPR | Ducking 속도·부드러움, 배경음 대비 대사 볼륨 유지력 |
| 주관 평가 | MUSHRA | 블라인드 청취 테스트 (14명) — 통계적 유의성 검증 |

## 디렉토리 구조
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
├── evaluation/                 # 청취 평가 (담당 트랙)
│   ├── webmushra/              #   MUSHRA 블라인드 테스트 UI
│   ├── results/                #   청취 결과 (gitignored)
│   └── rehearsal_checklist.md
│
├── tools/                      # MUSHRA·V3.5.x 평가 보조 도구
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
