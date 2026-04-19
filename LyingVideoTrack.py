from ultralytics import YOLO
from aiortc import MediaStreamTrack
import cv2
from av import VideoFrame
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO)


model = YOLO("yolo26n-pose.pt")


class LyingVideoTrack(MediaStreamTrack):

    kind = "video"

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.source_track = source_track
        self.last_infer_time = 0.0
        self.cached_items = []
        self.cached_frame = None
        self.infer_interval_sec = 0.12  # 약 8 FPS 정도만 추론

    def is_lying(self, kpts, confs, conf_th=0.5):
        # 필요한 keypoint 인덱스
        idx = [5, 6, 11, 12]

        # conf 체크
        for i in idx:
            if confs[i] < conf_th:
                return False  # 판단 불가

        # 어깨 중심
        shoulder = np.mean([kpts[5], kpts[6]], axis=0)
        # 골반 중심
        hip = np.mean([kpts[11], kpts[12]], axis=0)

        dx = abs(shoulder[0] - hip[0])
        dy = abs(shoulder[1] - hip[1])

        # 핵심 조건
        if dx > dy:
            return True  # 누움 (수평)
        else:
            return False  # 서있음 (수직)

    def draw_skeleton(self, kpts, confs, draw):
        # 🔴 keypoint 점
        for j, (x, y) in enumerate(kpts):
            if confs[j] < 0.5:
                continue
            cv2.circle(draw, (int(x), int(y)), 4, (0, 0, 255), -1)

        # 🔵 skeleton
        SKELETON = [
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        for i1, i2 in SKELETON:
            if confs[i1] < 0.5 or confs[i2] < 0.5:
                continue

            x1, y1 = map(int, kpts[i1])
            x2, y2 = map(int, kpts[i2])

            cv2.line(draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

    async def recv(self) -> VideoFrame:
        # print("@@ recv called")
        frame = await self.source_track.recv()

        # 입력 프레임 -> OpenCV BGR
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        if (
            self.cached_frame is None
            or (now - self.last_infer_time) >= self.infer_interval_sec
        ):
            self.last_infer_time = now

            results = model.predict(source=img, verbose=False, conf=0.35)

            # print(results)

            draw = img.copy()
            items = []

            if results and len(results) > 0:
                r = results[0]

                if r.keypoints is not None and r.boxes is not None:
                    person_count = min(len(r.keypoints.xy), len(r.boxes))

                    for i in range(person_count):
                        kpts = r.keypoints.xy[i]
                        confs = r.keypoints.conf[i]
                        box = r.boxes[i]

                        self.draw_skeleton(kpts, confs, draw)

                        state = self.is_lying(kpts, confs)
                        if state is True:
                            pose_text = "Lying"
                            pose_color = (0, 0, 255)  # 빨강
                        elif state is False:
                            pose_text = "Standing"
                            pose_color = (0, 255, 0)  # 초록
                        else:
                            pose_text = "Unknown"
                            pose_color = (0, 255, 255)  # 노랑

                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_name = model.names.get(cls_id, str(cls_id))

                        items.append(
                            {
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": round(conf, 4),
                                "bbox": [x1, y1, x2, y2],
                                "pose": pose_text,
                            }
                        )

                        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(
                            draw,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        (text_w, text_h), _ = cv2.getTextSize(
                            pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                        )

                        text_x = int((x1 + x2) / 2 - text_w / 2)
                        text_y = max(20, y1 - 8)

                        # 자세 텍스트 추가
                        cv2.putText(
                            draw,
                            pose_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            pose_color,
                            2,
                            cv2.LINE_AA,
                        )

                        print(f"{i}: {pose_text}")

                else:
                    print("keypoint 또는 boxes가 없음")

            self.cached_items = items
            self.cached_frame = draw
            logging.info("detections=%s", items[:3])

        # 추론하지 않는 프레임에도 마지막 결과가 그려진 최신 프레임 사용 가능
        output_bgr = self.cached_frame if self.cached_frame is not None else img

        # WebRTC 송출용 VideoFrame 생성
        new_frame = VideoFrame.from_ndarray(output_bgr, format="bgr24")

        # 매우 중요: 입력 프레임의 타임스탬프를 유지
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame
