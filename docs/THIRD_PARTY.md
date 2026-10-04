# 외부 모델·데이터 안내

저장소 자체 코드의 MIT 라이선스는 외부 모델·데이터·논문·영상의 이용 조건을 대신하지 않습니다.

## 데이터

| 자료 | 공식 출처 | 공개본에서의 처리 |
|---|---|---|
| AcousticRooms | [배포 저장소](https://github.com/facebookresearch/AcousticRooms), [CC BY 4.0](https://github.com/facebookresearch/AcousticRooms/blob/main/LICENSE) | 데이터 미포함. 출처 표기·라이선스 조건에 따라 별도 확보 |
| LIRIS-ACCEDE | [공식 배포처](https://liris-accede.ec-lyon.fr/) | 데이터 미포함. 배포처의 접근·이용 절차 확인 |
| COGNIMUSE | [관련 설계 기록](system-overview.md) | 평가 200클립 미포함. 원자료 이용 권한 별도 확인 |

AcousticRooms 기반 연구: Liu 외, *Hearing Anywhere in Any Environment*, CVPR 2025. [논문](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hearing_Anywhere_in_Any_Environment_CVPR_2025_paper.pdf) · [xRIR 구현](https://github.com/DragonLiu1995/xRIR_code).

## Essentia 모델

`model/autoEQ/pseudo_label/weights/`의 `msd-musicnn-1`, `deam-msd-musicnn-2`, `emomusic-msd-musicnn-2`, `muse-msd-musicnn-2`는 UPF Music Technology Group의 외부 pretrained 모델이며 이 프로젝트가 학습한 최종 Mood-EQ 가중치가 아닙니다. 함께 저장한 JSON에 저자·버전·다운로드 주소가 기록되어 있습니다. 외부 모델은 수정하지 않았습니다.

[공식 모델 안내](https://essentia.upf.edu/models.html)는 MTG 모델을 **CC BY-NC-SA 4.0**으로 안내합니다. 비상업적 이용과 동일조건변경허락 등 모델별 조건을 확인해야 하며 저장소의 MIT 라이선스를 근거로 상업적 사용을 허용한다고 해석하지 않습니다. [라이선스 조건](https://creativecommons.org/licenses/by-nc-sa/4.0/).

입력 영상·음원과 백본별 pretrained 모델도 각 제공처의 조건에 따라 준비합니다. 최종 3-seed 학습 체크포인트와 외부 영상·평가 음원은 포함하지 않습니다.
