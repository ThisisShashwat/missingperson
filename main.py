import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
import streamlink

from imutils.video import VideoStream

TWITCH_CHANNEL ="plastuchino"

FACECAM_X, FACECAM_Y = 0,0
FACECAM_W, FACECAM_H = 400, 260

missing_since = None
total_missing = 0.0


def get_stream_url(channel):
    streams = streamlink.streams(f"https://twitch.tv/{channel}")
    if not streams:
        raise RuntimeError("eee streams found")
    return streams["best"].to_url()

stream_url = get_stream_url(TWITCH_CHANNEL)

vs = VideoStream(stream_url).start()
time.sleep(1.0)

detector = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="face_detector.tflite"),
        running_mode=RunningMode.VIDEO,
    )
)

frame_count = 0
blackout = False

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame_count += 1

    facecam = frame[FACECAM_Y:FACECAM_Y + FACECAM_H, FACECAM_X:FACECAM_X + FACECAM_W]

    if blackout:
        facecam[:] = 0

    rgb = cv2.cvtColor(facecam, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(time.time() *1000))
    face_present = len(result.detections)  > 0

    if face_present:
        if missing_since is not None:
            total_missing += time.time() - missing_since
            missing_since = None
    else:
        if missing_since is None:
            missing_since = time.time()

    current_missing = total_missing - (time.time() - missing_since if missing_since else 0)
    print(f"missing: {current_missing:.1f}s")


    for d in result.detections:

        bb = d.bounding_box
        cv2.rectangle(facecam, (bb.origin_x, bb.origin_y),
                      (bb.origin_x + bb.width, bb.origin_y + bb.height), (0, 255, 0), 2)


    if frame_count % 30 == 0:
        print(f"frame {frame_count}, shape {frame.shape}")

    cv2.imshow("stream", facecam)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        blackout = not blackout

vs.stop()
cv2.destroyAllWindows()