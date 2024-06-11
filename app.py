from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the YOLOv8 model
logging.info("Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")
logging.info("Model loaded successfully!")


# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)


@app.route("/")
def index():
    logging.info("Rendering index page.")
    return render_template("index.html")


def gen():
    logging.info("Starting video capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame.")
            continue

        logging.debug("Performing object detection...")
        results = model(frame, stream=True)

        for result in results:
            logging.debug(f"Result: {result}")
            boxes = result.boxes.cpu().numpy()  # Convert to numpy array
            logging.debug(f"Detected {len(boxes)} objects.")

            for box in boxes:
                logging.debug(f"Box: {box}")
                if box.conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    color = getColours(cls)
                    label = f"{model.names[cls]} {box.conf[0]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )
                    logging.debug(f"Drawn box for {label} at ({x1}, {y1}, {x2}, {y2})")

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


@app.route("/video_feed")
def video_feed():
    logging.info("Starting video feed.")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False)
