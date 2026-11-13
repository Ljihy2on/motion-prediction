# 실시간 모션 분석 및 이동 경로 예측 시스템 (v1)

노트북 카메라 또는 합성 시뮬레이션을 통해 **색상 물체 추적 → Optical Flow → Kalman Filter 미래 위치 예측 → 위험구역 경고 → 예측 오차 측정**을 실행합니다.

## 설치

Python 3.10 이상 권장. 프로젝트 폴더에서:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py --mode sim
python main.py --mode camera --camera 0
python main.py --mode sim --save demo.mp4 --frames 600
python main.py --mode sim --headless --frames 300 --metrics results.csv
```

**Windows에서 `python` 명령이 안 되면 `py`를 사용하세요.** 카메라가 열리지 않으면 다른 앱의 카메라 점유 여부와 운영체제 권한을 확인하세요.

## 조작

- `q` 또는 `Esc`: 종료
- `r`: 칼만 필터, 이동 경로 및 측정 지표 초기화
- `f`: Optical Flow 화살표 표시/숨기기
- `p`: 예측 경로 표시/숨기기
- `z`: 위험구역 표시/숨기기
- `[` / `]`: 미래 예측 시간 감소/증가 (0.1~2.0초)
- `1`: 빨간색 물체 추적 (기본값, 빨간 공 또는 물건 사용)
- `2`: 파란색 물체 추적
- `3`: 초록색 물체 추적

## 화면 표시

- 청록색: 검출된 물체와 이동 이력
- 노란색 점선: 예측 경로
- 빨간 사각형: 가상 위험구역
- 좌상단: 검출 상태, 픽셀/초 속도, 예측 시간, Optical Flow, 예측 오차
- `PREDICTED ZONE ENTRY`: 현재 예측 경로가 지정 구역과 교차

**주의:** 카메라 화면의 속도는 `pixel/s`이며, 실제 m/s가 아닙니다. 2D 위험구역 경고는 실제 충돌 판정이 아닙니다. 카메라 모드의 예측 오차는 이전 예측을 나중의 관측값과 비교한 값으로, 색상 검출 오차도 포함합니다. 시뮬레이션 모드의 오차는 생성된 정답 좌표와 비교합니다.

## 설계

1. 카메라 모드: HSV 색상 임계값으로 물체 중심 검출. 배경이 같은 색이면 오검출될 수 있음.
2. 시뮬레이션 모드: 가상 물체를 움직이고 노이즈가 포함된 측정값 생성.
3. 칼만 필터: `[x,y,vx,vy]` 등속도 모델. 방향 전환 시 예측 오차가 커질 수 있음.
4. Optical Flow: Lucas-Kanade sparse optical flow로 주변 특징점의 이동 화살표 표시.
5. 위험구역: 화면 오른쪽에 고정된 예시 영역. 미래 경로의 샘플 지점이 영역에 들어가면 경고.
6. CSV: 관측 좌표, 추정 좌표, 미래 예측, 속도, 오차, 경고를 기록.

## 확장 방향

- HSV 범위 캘리브레이션 UI 및 다중 객체 추적
- 가속도/방향 전환 모델과 등속도 모델 성능 비교
- ROI 마우스 설정 및 시간-충돌(TTC) 추정
- 객체 탐지 모델(YOLO)과 결합

프로그램은 실제 웹캠 입력을 이 실행 환경에서 직접 테스트할 수 없으므로 카메라 권한/해상도/조명은 사용자 PC에서 확인해야 합니다.
