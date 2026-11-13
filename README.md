# Motion Prediction

웹캠 영상에서 물체를 추적하여, 지나온 경로와 예상 위치를 보여주는 시스템입니다.
Python과 OpenCV로 물체를 추적하고, 칼만 필터로 위치·속도를 추정해 잠시 뒤(기본 0.7초)의 위치를 예측합니다.
예상 경로가 화면에 지정한 가상 위험구역에 들어가면 경고를 표시합니다.

## 폴더 구성

| 폴더 | 내용 |
| --- | --- |
| [`motion_predictor_v1`](motion_predictor_v1) | HSV 색상 검출 기반 추적, Optical Flow, 칼만 필터 예측, 시뮬레이션 (`demo.mp4`, `sample_results.csv` 포함) |
| [`motion_predictor_v2`](motion_predictor_v2) | 마우스로 물체를 선택하는 ROI 템플릿 매칭 추적 추가, 기존 색상 모드 유지 |

## 실행 방법

```bash
cd motion_predictor_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --mode camera --camera 1   # 웹캠
python main.py --mode sim                 # 시뮬레이션
```

조작키, 실행 옵션 등 자세한 내용은 각 폴더의 README를 참고하세요.

## 개발 환경

- Python 3, OpenCV 4.8 이상, NumPy 1.24 이상
- MacBook 내장 웹캠
