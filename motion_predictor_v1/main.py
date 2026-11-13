"""Camera/simulation motion tracking, optical flow and Kalman prediction."""
import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

COLORS = {
    'red': [((0, 90, 75), (12, 255, 255)), ((170, 90, 75), (179, 255, 255))],
    'blue': [((95, 90, 65), (130, 255, 255))],
    'green': [((38, 75, 60), (85, 255, 255))],
}


def make_filter(x, y):
    k = cv2.KalmanFilter(4, 2)
    k.transitionMatrix = np.eye(4, dtype=np.float32)
    k.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    k.processNoiseCov = np.diag([1.5, 1.5, 25, 25]).astype(np.float32)
    k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 12
    k.errorCovPost = np.eye(4, dtype=np.float32) * 100
    k.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return k


def detect_color(frame, name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLORS[name]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 120:
        return None
    (x, y), radius = cv2.minEnclosingCircle(c)
    return (float(x), float(y), float(radius))


def simulated_frame(t, width, height, rng):
    """Deterministic moving target with direction changes and measurement noise."""
    frame = np.full((height, width, 3), (27, 35, 43), np.uint8)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (44, 52, 62), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (44, 52, 62), 1)
    x = width * (0.5 + 0.36 * math.sin(t * 0.67))
    y = height * (0.5 + 0.28 * math.sin(t * 1.04 + 0.6))
    cv2.circle(frame, (int(x), int(y)), 19, (35, 45, 235), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(x - 5), int(y - 6)), 5, (135, 170, 255), -1, cv2.LINE_AA)
    measurement = (x + rng.normal(0, 2.5), y + rng.normal(0, 2.5), 19.0)
    return frame, measurement, (x, y)


def inside(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_flow(frame, prev_gray, gray, prev_pts):
    if prev_gray is None or prev_pts is None or len(prev_pts) < 5:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=.015, minDistance=10)
        return pts
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if nxt is None or status is None:
        return None
    old_good = prev_pts[status.flatten() == 1].reshape(-1, 2)
    new_good = nxt[status.flatten() == 1].reshape(-1, 2)
    for (ox, oy), (nx, ny) in zip(old_good[::2], new_good[::2]):
        if np.hypot(nx - ox, ny - oy) >= 0.6:
            cv2.arrowedLine(frame, (int(ox), int(oy)), (int(nx), int(ny)), (255, 190, 90), 1, tipLength=.35)
    return new_good.reshape(-1, 1, 2).astype(np.float32) if len(new_good) >= 12 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sim', 'camera'], default='sim')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--frames', type=int, default=0, help='0 = unlimited')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--save', default='', help='MP4 output path')
    parser.add_argument('--metrics', default='', help='CSV output path')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.headless and args.mode == 'camera' and args.frames == 0:
        parser.error('headless camera mode requires --frames')
    if args.fps <= 0 or args.width < 160 or args.height < 120:
        parser.error('fps must be positive and image at least 160x120')
    rng = np.random.default_rng(args.seed)
    cap = None
    if args.mode == 'camera':
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError('카메라를 열 수 없습니다. --camera 번호와 카메라 권한을 확인하세요.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    writer = None
    csv_file = None
    csv_writer = None
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.metrics, 'w', newline='', encoding='utf-8-sig')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame','time_s','observed_x','observed_y','estimate_x','estimate_y',
                             'velocity_px_s','prediction_x','prediction_y','error_px','warning'])
    kalman = None
    history = deque(maxlen=100)
    pending = deque()
    prev_gray = None
    prev_pts = None
    color = 'red'
    show_flow = show_prediction = show_zone = True
    horizon = .7
    frame_id = 0
    last_time = None
    recent_errors = deque(maxlen=200)
    import time
    try:
        while args.frames == 0 or frame_id < args.frames:
            if args.mode == 'sim':
                t = frame_id / args.fps
                frame, detection, ground_truth = simulated_frame(t, args.width, args.height, rng)
                dt = 1 / args.fps
            else:
                ok, frame = cap.read()
                if not ok:
                    print('카메라에서 프레임을 읽지 못했습니다.')
                    break
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (args.width, args.height))
                now = time.monotonic()
                dt = 1 / args.fps if last_time is None else float(np.clip(now-last_time, .005, .25))
                last_time = now
                t = now
                detection = detect_color(frame, color)
                ground_truth = None
            h, w = frame.shape[:2]
            rect = (int(w*.64), int(h*.28), int(w*.92), int(h*.72))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute flow on clean input frames, not on annotated frames.
            if show_flow:
                prev_pts = draw_flow(frame, prev_gray, gray, prev_pts)
            else:
                prev_pts = None
            prev_gray = gray
            if show_zone:
                overlay = frame.copy()
                cv2.rectangle(overlay, rect[:2], rect[2:], (40, 55, 130), -1)
                frame = cv2.addWeighted(overlay, .16, frame, .84, 0)
                cv2.rectangle(frame, rect[:2], rect[2:], (75, 85, 230), 2)
                cv2.putText(frame, 'VIRTUAL RISK ZONE', (rect[0], rect[1]-9),
                            cv2.FONT_HERSHEY_SIMPLEX, .46, (100, 120, 255), 1)
            if detection is not None and kalman is None:
                kalman = make_filter(detection[0], detection[1])
            estimate = None
            velocity = 0.0
            prediction = None
            warning = False
            if kalman is not None:
                kalman.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],
                                                     [0,0,1,0],[0,0,0,1]], np.float32)
                kalman.predict()
                if detection is not None:
                    measured = np.array([[detection[0]], [detection[1]]], np.float32)
                    kalman.correct(measured)
                state = kalman.statePost.flatten()
                ex, ey, vx, vy = [float(v) for v in state]
                estimate = (ex, ey)
                velocity = math.hypot(vx, vy)
                prediction = (ex + vx*horizon, ey + vy*horizon)
                history.append((int(ex), int(ey)))
                if show_prediction:
                    pts = [(int(ex + vx*horizon*i/16), int(ey + vy*horizon*i/16)) for i in range(17)]
                    for a,b in zip(pts[::2], pts[1::2]):
                        cv2.line(frame, a, b, (20, 220, 255), 2)
                    cv2.circle(frame, tuple(map(int, prediction)), 7, (20, 220, 255), 2)
                warning = show_zone and any(inside(ex + vx*horizon*i/20,
                                                    ey + vy*horizon*i/20, rect) for i in range(21))
                if len(history) > 1:
                    cv2.polylines(frame, [np.array(history, np.int32)], False, (245, 220, 55), 2)
                cv2.circle(frame, (int(ex),int(ey)), 5, (255,255,0), -1)
            if detection is not None:
                dx, dy, radius = detection
                cv2.circle(frame, (int(dx),int(dy)), int(radius), (70,255,100), 2)
                cv2.circle(frame, (int(dx),int(dy)), 3, (70,255,100), -1)
            # Score a prediction made in a previous frame against future ground truth/observation.
            actual = ground_truth if ground_truth is not None else (detection[:2] if detection else None)
            while pending and pending[0][0] <= frame_id:
                _, px, py = pending.popleft()
                if actual is not None:
                    recent_errors.append(math.hypot(px-actual[0], py-actual[1]))
            if prediction is not None:
                future_frame = frame_id + max(1, int(round(horizon / dt)))
                pending.append((future_frame, prediction[0], prediction[1]))
            mae = float(np.mean(recent_errors)) if recent_errors else float('nan')
            cv2.rectangle(frame, (0,0), (min(w, 420), 100), (12,20,30), -1)
            labels = [f'MODE: {args.mode.upper()}  TRACK: {color.upper()}  DETECT: {"YES" if detection else "NO"}',
                      f'SPEED: {velocity:.1f} px/s    PREDICT: {horizon:.1f}s',
                      f'PREDICTION ERROR: {mae:.1f} px' if not math.isnan(mae) else 'PREDICTION ERROR: collecting...',
                      'q:quit r:reset f:flow p:prediction z:zone [ ]:horizon']
            for i, line in enumerate(labels):
                cv2.putText(frame, line, (10, 20+i*24), cv2.FONT_HERSHEY_SIMPLEX,
                            .47 if i<3 else .4, (210,240,245), 1, cv2.LINE_AA)
            if warning:
                cv2.putText(frame, 'PREDICTED ZONE ENTRY', (max(5,w//2-160), h-24),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (40,65,255), 2, cv2.LINE_AA)
            if csv_writer:
                csv_writer.writerow([frame_id,round(frame_id/args.fps,4) if args.mode=='sim' else round(t,4),
                                     detection[0] if detection else '', detection[1] if detection else '',
                                     estimate[0] if estimate else '', estimate[1] if estimate else '',
                                     velocity, prediction[0] if prediction else '',
                                     prediction[1] if prediction else '', mae if recent_errors else '', int(warning)])
            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'),
                                             args.fps, (w,h))
                    if not writer.isOpened():
                        raise RuntimeError('동영상 파일을 열 수 없습니다: ' + args.save)
                writer.write(frame)
            if not args.headless:
                cv2.imshow('Motion Prediction v1', frame)
                key = cv2.waitKey(max(1,int(1000/args.fps)) if args.mode=='sim' else 1) & 0xff
                if key in (27, ord('q')):
                    break
                if key == ord('r'):
                    kalman = None; history.clear(); pending.clear(); recent_errors.clear()
                elif key == ord('f'): show_flow = not show_flow
                elif key == ord('p'): show_prediction = not show_prediction
                elif key == ord('z'): show_zone = not show_zone
                elif key == ord('['): horizon = max(.1, round(horizon-.1, 1))
                elif key == ord(']'): horizon = min(2., round(horizon+.1, 1))
                elif key in (ord('1'),ord('2'),ord('3')):
                    color = {'1':'red','2':'blue','3':'green'}[chr(key)]
                    kalman = None; history.clear(); pending.clear(); recent_errors.clear()
            frame_id += 1
        print(f'완료: {frame_id}프레임 / 평균 예측 오차: {float(np.mean(recent_errors)):.2f}px' if recent_errors
              else f'완료: {frame_id}프레임 / 예측 오차 측정 데이터 부족')
    finally:
        if cap is not None: cap.release()
        if writer is not None: writer.release()
        if csv_file is not None: csv_file.close()
        if not args.headless: cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
