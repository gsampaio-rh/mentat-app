from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
from pyzbar import pyzbar
import numpy as np
import logging
import argparse
import os
import json

app = Flask(__name__)
icon_counters = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection with Webcam Stream"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Choose the YOLOv8 model to use",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


def set_logging_level(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(level=numeric_level)


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


def detect_objects(frame, model):
    logging.debug("Performing object detection...")
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        logging.debug(f"Detected {len(boxes)} objects.")
        for box in boxes:
            if box.conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                color = getColours(cls)
                label = f"{model.names[cls]} {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )
                logging.debug(f"Drawn box for {label} at ({x1}, {y1}, {x2}, {y2})")
    return frame


def overlay_image(frame, overlay_img, position, icon_name):
    x, y = position
    h, w = overlay_img.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    # Adjust overlay dimensions to fit within the frame
    if x + w > frame_w:
        w = frame_w - x
        overlay_img = overlay_img[:, :w]
    if y + h > frame_h:
        h = frame_h - y
        overlay_img = overlay_img[:h]

    # Resize overlay image to fit exactly the region in the frame
    overlay_img = cv2.resize(overlay_img, (w, h))

    # Draw icon name above the overlay image
    font_scale = 0.6
    font_thickness = 2
    text_size = cv2.getTextSize(
        icon_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y - 10 if y - 10 > text_size[1] else y + h + text_size[1] + 10

    cv2.putText(
        frame,
        icon_name,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        font_thickness,
    )

    # Overlay the image
    overlay_img_gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(overlay_img_gray, 1, 255, cv2.THRESH_BINARY)
    frame_bg = cv2.bitwise_and(
        frame[y : y + h, x : x + w],
        frame[y : y + h, x : x + w],
        mask=cv2.bitwise_not(mask),
    )
    img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
    frame[y : y + h, x : x + w] = cv2.add(frame_bg, img_fg)


def decode_qr_code(frame, overlay_images, qr_mappings):
    global icon_counters
    decoded_objects = pyzbar.decode(frame)
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(
                np.array([point for point in points], dtype=np.float32)
            )
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        n = len(hull)
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 255, 0), 3)

        (x, y, w, h) = obj.rect
        qr_data = obj.data.decode("utf-8")
        logging.debug(f"QR Code data: {qr_data}")

        mapping = qr_mappings.get(qr_data)
        if mapping is not None:
            overlay_filename = mapping.get("overlay_image")
            icon_name = mapping.get("icon_name", "Unknown")
            overlay_img = overlay_images.get(overlay_filename)
            if overlay_img is not None:
                logging.debug(f"Found overlay image for QR code: {qr_data}")
                resized_overlay_img = cv2.resize(overlay_img, (w, h))
                overlay_image(frame, resized_overlay_img, (x, y), icon_name)
                icon_counters[icon_name] = (
                    icon_counters.get(icon_name, 0) + 1
                )  # Increment counter for each detected icon
            else:
                logging.warning(f"Overlay image file not found: {overlay_filename}")
        else:
            logging.warning(f"No overlay mapping found for QR code: {qr_data}")

    return frame, decoded_objects


def load_overlay_images(image_directory):
    overlay_images = {}
    for filename in os.listdir(image_directory):
        if filename.startswith("overlay_"):
            overlay_images[filename] = cv2.imread(
                os.path.join(image_directory, filename)
            )
    return overlay_images


def load_qr_mappings(mapping_file):
    with open(mapping_file, "r") as f:
        return json.load(f)


def gen(detection_enabled, qr_detection_enabled, overlay_images, qr_mappings):
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

        if detection_enabled:
            frame = detect_objects(frame, model)

        if qr_detection_enabled:
            frame, _ = decode_qr_code(frame, overlay_images, qr_mappings)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


@app.route("/")
def index():
    logging.info("Rendering index page.")
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    logging.info("Received request for /video_feed")
    detection_enabled = request.args.get("detection", "false").lower() == "true"
    qr_detection_enabled = request.args.get("qr_detection", "false").lower() == "true"
    overlay_images = load_overlay_images("images/")
    qr_mappings = load_qr_mappings("qr_mappings.json")
    logging.info(
        f"Starting video feed with detection enabled: {detection_enabled}, QR detection enabled: {qr_detection_enabled}"
    )
    return Response(
        gen(detection_enabled, qr_detection_enabled, overlay_images, qr_mappings),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/icon_counters")
def get_icon_counters():
    return jsonify(icon_counters)


if __name__ == "__main__":
    args = parse_args()
    set_logging_level(args.log)
    logging.info(f"Loading YOLOv8 model: {args.model}...")
    model = YOLO(args.model)
    logging.info("Model loaded successfully!")
    logging.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=8000, debug=False)
