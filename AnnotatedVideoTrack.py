from ultralytics import YOLO
from aiortc import MediaStreamTrack
import cv2
from av import VideoFrame
import logging
import time

logging.basicConfig(level=logging.INFO)


model = YOLO("yolov8n.pt")


class AnnotatedVideoTrack(MediaStreamTrack):
    """
    원격에서 들어온 비디오 트랙을 받아
    YOLO 추론 + 박스 그리기 후
    새 비디오 프레임으로 다시 송출하는 트랙
    """

    kind = "video"

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.source_track = source_track
        self.last_infer_time = 0.0
        self.cached_items = []
        self.cached_frame = None
        self.infer_interval_sec = 0.12  # 약 8 FPS 정도만 추론

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
                if r.boxes is not None:
                    for box in r.boxes:
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
