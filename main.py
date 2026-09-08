import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
import streamlink

TWITCH_CHANNEL ="plastuchino"

FACECAM_X, FACECAM_Y = 0,0
FACECAM_W, FACECAM_H = 400, 260


def get_stream_url(channel):
    streams = streamlink.streams(f"https://twitch.tv/{channel}")
    if not streams:
        raise RuntimeError("eee streams found")
    return streams["best"].to_url()

stream_url = get_stream_url(TWITCH_CHANNEL)

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("cant stream wtf")

detector = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="face_detector.tflite"),
        running_mode=RunningMode.VIDEO,
    )
)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ughh?")
        break
    frame_count += 1

    facecam = frame[FACECAM_Y:FACECAM_Y + FACECAM_H, FACECAM_X:FACECAM_X + FACECAM_W]

    rgb = cv2.cvtColor(facecam, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(time.time() *1000))
    face_present = len(result.detections)  > 0

    print(face_present)

    for d in result.detections:

        bb = d.bounding_box
        cv2.rectangle(facecam, (bb.origin_x, bb.origin_y),
                      (bb.origin_x + bb.width, bb.origin_y + bb.height), (0, 255, 0), 2)


    if frame_count % 30 == 0:
        print(f"frame {frame_count}, shape {frame.shape}")

    cv2.imshow("stream", facecam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()