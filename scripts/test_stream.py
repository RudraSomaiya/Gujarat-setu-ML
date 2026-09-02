import cv2
import os

# Connect to the RTSP stream directly
rtsp_url = "rtsp://103.250.160.189:8554/stream/cam06"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print(f"Opening stream: {rtsp_url}")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open stream!")
    exit(1)

print("Stream opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame.")
        break

    cv2.imshow("Direct Stream Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
