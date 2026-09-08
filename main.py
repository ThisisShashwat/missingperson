import cv2
import mediapipe as mp

import streamlink

TWITCH_CHANNEL ="plastuchino"

def get_stream_url(channel):
    streams = streamlink.streams(f"https://twitch.tv/{channel}")
    if not streams:
        raise RuntimeError("eee streams found")
    return streams["best"].to_url()

stream_url = get_stream_url(TWITCH_CHANNEL)

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("cant stream wtf")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ughh?")
        break
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"frame {frame_count}, shape {frame.shape}")

    cv2.imshow("stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()